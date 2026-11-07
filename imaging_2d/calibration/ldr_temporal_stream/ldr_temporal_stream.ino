const uint8_t LDR_PIN = A0;
const unsigned long SAMPLE_INTERVAL_MS = 20;

unsigned long nextSample = 0;

void setup() {
  Serial.begin(115200);
  analogReference(DEFAULT);
  pinMode(LDR_PIN, INPUT);

  delay(300);
  nextSample = millis();

  Serial.println(F("READY,TEMPORAL_STREAM"));
}

void loop() {
  unsigned long now = millis();

  if ((long)(now - nextSample) >= 0) {
    int adc = analogRead(LDR_PIN);

    Serial.print(F("DATA,"));
    Serial.print(now);
    Serial.print(",");
    Serial.println(adc);

    nextSample += SAMPLE_INTERVAL_MS;
  }
}
