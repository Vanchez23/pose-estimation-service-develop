# POSE ESTIMATION SERVICE #

Сервис для определения ключевых точек.

Данный сервис взаимодействует с сервисом хранилища (storage-service). Каждые N секунд  
(длительность задается в настройках) к storage-service   
отправляется запрос, в котором указывается какое количество кадров прислать в ответ на обработку.  

Ответ от storage-service приходит в виде списка словарей:   
[{"task_id": <task_id>, "frame_id": <frame_id>, "path": \<path\>}, ...],  
где 
- \<task_id\> - идентификатор задачи (int);  
- \<frame_id\> - идентификатор кадра (int);  
- \<path\> - относительный путь до кадра, имеет вид "<task_id>/images/<folder_name>/<frame_id>.jpg" (str). 


После получения ответа данный сервис обрабатывает все кадры и отправляет результат обработки storage-service
в формате:  
[{"frame_id": <frame_id>, "duration": <duration>, "results": \<results\>}, ...],  
где 
- \<frame_id\> - идентификатор кадра (int);  
- \<duration\> - время обработки кадра, сек. (float);  
- \<results\> - результат обработки кадра (подробное описание ниже) (list).  
  
\<results\> -  представляет список словарей вида:  
[\<point_name\>: {  
- "x": \<coordinate x\>,  
- "y": \<coordinate y\>,  
- "proba": \<proba\>,  
},...], где  
- \<point_name\> - название точки из списка: ['l_sho', 'r_sho', 'l_elb', 'r_elb', 'l_wri', 'r_wri'] (str).  
- \<x\> - индекс пикселя по ширине в относительных координатах (float);  
- \<y\> - индекс пикселя по высоте в относительных координатах (float);  
- \<proba\> - вероятность предсказания (float). 

### Настройка сервиса ###
1. Установка 
```
git clone https://gitlab.sch.ocrv.com.rzd/cv-working-time-regulation/pose-estimation-service.git
cd pose-estimation-service
```
2. Создание виртулаьного окружения 
```
virtualenv -p python3.6 venv
. venv/bin/activate
```
3. Создать локальные настройки и изменить на необходимые 
```
. venv/bin/activate
pip install -r requirements.txt
cp settings/local.py.default settings/local.py
```
4. Установить библиотеку tf_simple_human_pose по аналогии с Dockerfile:
```
https://gitlab.sch.ocrv.com.rzd/public-repos/docker-images/blob/tf_human_pose_cpu
```
5. Скачать файл модели
```
scp developer@172.22.100.219:/home/developer/service_models/pose_estimation/* models/
```
### Запуск сервиса ###
```
. venv/bin/activate
python manage.py run
```
### Запуск тестов ###
```
. venv/bin/activate
python manage.py test [--converage]
```
 



