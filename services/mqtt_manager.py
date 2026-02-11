import time
import json
import random
import string
import streamlit as st
import paho.mqtt.client as mqtt
import paho.mqtt.publish as publish

def get_mqtt_config():
    try:
        return st.secrets["mqtt"]
    except Exception:
        # Fallback/Default if secrets missing (though user requirement says use secrets)
        return {
            "broker": "data.metr.dk",
            "port": 8080,
            "username": "pxi",
            "password": "eQzX7uefeTtnfBdrg1nIYTlAtZiR2K5o",
            "transport": "websockets"
        }



# User-defined topic mapping
TOPIC_MAP = {
    "slshome/sensor/solar_pv1_power/state": "pv",
    "slshome/sensor/solar_load_l1_power/state": "load1",
    "slshome/sensor/solar_load_l2_power/state": "load2",
    "slshome/sensor/solar_load_l3_power/state": "load3",
    "slshome/sensor/solar_battery_soc/state": "soc",
    "slshome/sensor/solar_battery_power/state": "bat",
    "slshome/sensor/solar_total_grid_power/state": "grid",
}

# Validated Singleton Pattern for Streamlit Backgound Threads
@st.cache_resource
class MQTTService:
    def __init__(self):
        self.client = None
        self.status = "Initializing..."
        self.data_cache = {}
        self.last_message = {"topic": "None", "payload": "None", "time": "Never"}
        self._setup_client()

    def _setup_client(self):
        conf = get_mqtt_config()
        
        # Determine Transport (Default to WebSockets 8080 as it's more firewall friendly)
        transport = conf.get("transport", "websockets")
        port = conf.get("port", 8080)
        
        # Fix: If user secrets specify TCP port (1883) but we are using WebSockets, 
        # force the standard WS port (8080) to avoid handshake errors.
        if transport == "websockets" and port == 1883:
            port = 8080


        # Randomize Client ID to avoid conflicts
        rnd_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        client_id = f"streamlit_backend_{rnd_id}"

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id, transport=transport)
        self.client.username_pw_set(conf["username"], conf["password"])
        
        if transport == "websockets":
            self.client.ws_set_options(path="/mqtt")

        # Callbacks must update THIS instance, not session_state
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect
        self.client.on_subscribe = self.on_subscribe
        # self.client.on_log = self.on_log # Disable verbose logging to status

        try:
            self.client.connect(conf["broker"], port, keepalive=60)
            self.client.loop_start()
            self.status = f"Connecting to {conf['broker']}:{port} ({transport})..."
        except Exception as e:
            self.status = f"Setup Error: {e}"

    # def on_log(self, client, userdata, level, buf):
    #    pass 

    def on_subscribe(self, client, userdata, mid, reason_code_list, properties=None):
        # Confirm subscription success
        if hasattr(reason_code_list[0], 'is_failure') and reason_code_list[0].is_failure:
             print(f"Subscribe Failed: {reason_code_list}")
             self.status = f"Sub Failed: {reason_code_list}"
        else:
             print(f"Subscribed (MID: {mid}) RC: {reason_code_list}")


    def on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            self.status = "Connected"
            # Subscribe to mapped topics
            for topic in TOPIC_MAP.keys():
                client.subscribe(topic)
        else:
            self.status = f"Connection Failed (RC: {rc})"

    def on_disconnect(self, client, userdata, flags, rc, properties=None):
        self.status = f"Disconnected (RC: {rc})"

    def on_message(self, client, userdata, msg):
        try:
            topic = msg.topic
            payload = msg.payload.decode()
            
            # Debug: Store last message (ANY topic)
            self.last_message = {
                "topic": topic,
                "payload": payload,
                "time": time.strftime("%H:%M:%S")
            }

            val = 0.0
            
            # Parsing logic
            if payload.startswith("{"):
                try:
                    data = json.loads(payload)
                    val = float(data.get("state", data.get("value", 0)))
                except:
                    pass
            else:
                try:
                    val = float(payload)
                except:
                    pass

            if topic in TOPIC_MAP:
                key = TOPIC_MAP[topic]
                self.data_cache[key] = val
                
                # Load Calc
                if key.startswith("load"):
                    l1 = self.data_cache.get("load1", 0)
                    l2 = self.data_cache.get("load2", 0)
                    l3 = self.data_cache.get("load3", 0)
                    self.data_cache["load"] = l1 + l2 + l3
                    
        except Exception as e:
            print(f"Msg Error: {e}")


def get_client() -> mqtt.Client:
    """
    Returns a connected MQTT client from the singleton service.
    Compatibility wrapper for legacy code.
    """
    return get_service().client

# Accessor functions that use the singleton
def get_service():
    return MQTTService()

def start_mqtt_listener():
    get_service() # Ensures initialization

def get_latest_data():
    return get_service().data_cache

def get_connection_status():
    return get_service().status

def get_last_message():
    return get_service().last_message




def publish_message(topic, payload, retain=True, qos=1):
    """
    Publishes a message using the singleton client.
    Returns (success: bool, error_msg: str)
    """
    service = get_service()
    if not service.client:
        return False, "No MQTT client available"
    
    try:
        if isinstance(payload, (dict, list)):
            payload = json.dumps(payload, ensure_ascii=False)
        else:
            payload = str(payload)
            
        service.client.publish(topic, payload, qos=qos, retain=retain)
        return True, None
    except Exception as e:
        return False, str(e)


def get_retained(topic, timeout=1.5):
    """
    One-off read of a retained message. 
    Does NOT use the persistent session client to avoid callback conflicts,
    or we could use a separate one-shot client.
    Existing logic used a separate client for clean one-shot reads.
    """
    conf = get_mqtt_config()
    out = {}

    def _on_msg(c, u, m):
        out["p"] = m.payload.decode()

    # Use a fresh client for this specific request to ensure clean state
    # (or could attach temporary callback to session client, but risk race conditions)
    transport = conf.get("transport", "websockets")
    port = conf.get("port", 8080)
    
    cli = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, transport=transport)
    cli.username_pw_set(conf["username"], conf["password"])
    
    if transport == "websockets":
        cli.ws_set_options(path="/mqtt")
        
    cli.on_message = _on_msg

    try:
        cli.connect(conf["broker"], port, 60)
        cli.subscribe(topic)
        cli.loop_start()

        t0 = time.time()
        while "p" not in out and time.time() - t0 < timeout:
            time.sleep(0.05)

        cli.loop_stop()
        cli.disconnect()
        return out.get("p")
    except Exception:
        return None
