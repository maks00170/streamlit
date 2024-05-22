# Установка
```pip install -r requirements.txt```

# Запуск
```
streamlit run main.py
```

For my configuration:
- 16GB RAM
- 8th Gen Intel(R) Core(TM) i7-8750H

## Топ версия OpenVino
**Тесты на одном изображении**
 **Версия** | **Время** 
---|---
Default Yolo | 84.9 ms ± 2.52 ms per loop
OpenVino Yolo | 44.5 ms ± 2.6 ms per loop

Запуск с OpenVino

```python
from ultralytics import YOLO

ov_model = YOLO("weights/best_3000_openvino_model/", task='detect')
```