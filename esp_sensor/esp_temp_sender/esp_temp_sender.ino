#include <OneWire.h>
#include <DallasTemperature.h>

// ── เลือก board ──────────────────────────────────────
#ifdef ESP32
  #include <WiFi.h>
  #include <HTTPClient.h>
  #include <WiFiClientSecure.h>
#else
  #include <ESP8266WiFi.h>
  #include <ESP8266HTTPClient.h>
  #include <WiFiClientSecureBearSSL.h>
#endif

// ── ตั้งค่า WiFi ─────────────────────────────────────
const char* WIFI_SSID = "Meepu";
const char* WIFI_PASS = "0889144715";

// ── URL ของ Render (ออนไลน์ตลอด ไม่ต้องเปิด PC) ──────
const char* SERVER_URL = "https://betta-fish-app.onrender.com/api/temperature";

// ── ขา Data ของ DS18B20 ──────────────────────────────
#ifdef ESP32
  #define TEMP_PIN 4   // GPIO4
#else
  #define TEMP_PIN 2  // D2 บน ESP8266
#endif

// ── ส่งข้อมูลทุกกี่วินาที ────────────────────────────
#define SEND_INTERVAL 2000   // 2 วินาที

OneWire oneWire(TEMP_PIN);
DallasTemperature sensors(&oneWire);

void setup() {
  Serial.begin(115200);
  pinMode(TEMP_PIN, INPUT_PULLUP);
  sensors.begin();

  Serial.print("\nกำลังเชื่อมต่อ WiFi");
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500); Serial.print(".");
  }
  Serial.println("\nเชื่อมต่อสำเร็จ! IP: " + WiFi.localIP().toString());
}

void loop() {
  sensors.requestTemperatures();
  float temp = sensors.getTempCByIndex(0);

  if (temp == DEVICE_DISCONNECTED_C) {
    Serial.println("[ERROR] ไม่พบเซ็นเซอร์ — ตรวจสอบสายต่อ");
    delay(5000);
    return;
  }

  Serial.println("อุณหภูมิ: " + String(temp, 1) + " °C");
  sendTemperature(temp);
  delay(SEND_INTERVAL);
}

void reconnectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;
  Serial.print("[WARN] WiFi หลุด กำลัง reconnect");
  WiFi.disconnect();
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  int tries = 0;
  while (WiFi.status() != WL_CONNECTED && tries < 20) {
    delay(500); Serial.print("."); tries++;
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println(" เชื่อมต่อสำเร็จ!");
  } else {
    Serial.println(" ล้มเหลว จะลองใหม่รอบหน้า");
  }
}

void sendTemperature(float temp) {
  reconnectWiFi();
  if (WiFi.status() != WL_CONNECTED) return;

  String url  = String(SERVER_URL);
  String body = "{\"temperature\":" + String(temp, 1) + "}";

#ifdef ESP32
  WiFiClientSecure client;
  client.setInsecure();  // ข้าม SSL verify (self-signed cert)
  HTTPClient http;
  http.begin(client, url);
#else
  BearSSL::WiFiClientSecure client;
  client.setInsecure();
  HTTPClient http;
  http.begin(client, url);
#endif

  http.addHeader("Content-Type", "application/json");
  int code = http.POST(body);

  if (code == 200) {
    Serial.println("ส่งข้อมูลสำเร็จ ✓");
  } else {
    Serial.println("[ERROR] HTTP " + String(code));
  }
  http.end();
}
