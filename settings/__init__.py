# -*- coding: utf-8 -*-
import logging.config
from .base import *  # noqa

try:
    from .local import *  # noqa
except ImportError:
    exit('Необходимо заполнить настройки в файл settings/local.py.'
         '\nДля создание заготовки файла используйте команду: "cp settings/local.py.default settings/local.py"')

try:
    logging.config.dictConfig(LOGGING)  # noqa: F405
except NameError:
    exit('Define LOGGING in settings')
