# Reflective Single-Pixel Imaging — Calibration

This project uses a projector to illuminate a reflective target and a light-dependent resistor (LDR) to measure the returned light. A Python program controls the projected patterns, while an Arduino UNO records the LDR voltage and sends measurements to the computer over USB serial.

This README covers the hardware setup and calibration workflow: electrical response, projector alignment, temporal response, measurement repeatability, spatial response, and acquisition timing.

The scripts described below are restored versions of the calibration programs. They preserve the documented measurement methods but have not yet been revalidated on the hardware after restoration. Recorded settings are starting points for reproducing the setup, rather than universal calibration values.

<!-- IMAGE: 在这里放实验装置全景照片，标出 projector、reflective target、LDR、Arduino 和 computer，以及照明和反射光路。建议路径：docs/images/setup_overview.jpg。照片添加后取消下一行注释。 -->
<!-- ![Reflective imaging calibration setup](docs/images/setup_overview.jpg) -->

## Parts List

The quantities below describe one detector channel.

| Component | Quantity | Specification | Purchase link |
|---|---:|---|---|
| Projector | 1 | XGIMI Z7 | https://xgimi.com.au/product/xgimi-elfin-flip-laser-portable-projector-1600-iso-rgb-triple-laser-google-tv-netflix-harman-kardon-audio/ |
| Arduino | 1 | Arduino UNO R3 Board | https://www.jaycar.com.au/duinotech-uno-r3-main-board/p/XC4410  |
| LDR | 3 | Large Light Dependent Resistor (LDR) | https://www.jaycar.com.au/large-light-dependent-resistor-ldr/p/RD3485 |
| Resistor | 3 | 22 kΩ | https://www.jaycar.com.au/22k-ohm-0-5-watt-metal-film-resistors-pack-of-8/p/RR0604 |
| Tracing Paper | 1 | | https://www.officeworks.com.au/shop/officeworks/p/studymate-a4-tracing-pad-50-leaf-orange-smtrce50r?region_id=GTYP6H&regionIdSet=true |
| Foam Board | 3 |  | https://www.officeworks.com.au/shop/officeworks/p/a3-foam-board-5mm-white-ljpcfba3we |
