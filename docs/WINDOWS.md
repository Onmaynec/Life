# Windows: проверка сборки и системной рамки

Сборка автономного пакета выполняется workflow `.github/workflows/windows-build.yml` на Windows runner. Это каталог с `Life.exe` и встроенными библиотеками, а не одиночный файл. Выгрузка из Actions — **артефакт для тестирования**, пока не завершена приёмка на Windows 10/11.

Для локальной сборки на Windows с установленным Python 3.12:

```powershell
python -m pip install -r requirements.txt 'pyinstaller>=6,<7'
python -m PyInstaller --noconfirm --clean --onedir --windowed --name Life --hidden-import matplotlib.backends.backend_tkagg --hidden-import mpl_toolkits.mplot3d main.py
Start-Process -FilePath '.\dist\Life\Life.exe' -ArgumentList '--smoke-test' -Wait -PassThru
```

Распакованный каталог нельзя сокращать до одного `.exe`: нужны файлы в `_internal`. Запуск из исходников: `python main.py`; встроенная диагностика: `python main.py --smoke-test`.

Диагностика проверяет создание мира, сохранение/открытие, каталог RLE, PNG, CSV и пользовательские настройки. Она не заменяет проверку интерактивного графического интерфейса и корректности заголовка на разных версиях ОС.

Системная рамка оформляется через DWM, если ОС поддерживает нужные атрибуты. При неудаче Life оставляет родные системные кнопки и рамку Windows, без собственных накладок.
