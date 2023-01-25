import traceback
from flask import Blueprint, request, Response, Flask
import json, sys
import time
import requests
from colorama import Fore
import threading
import os
from pathlib import Path
import logging
from functools import wraps
from gym import logger, spaces
from gym.utils import seeding
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# Define the blueprint: 'train', set its url prefix: app.url/test

def create_logger(name):
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))

    logger = logging.getLogger(name)
    if logger.hasHandlers():
        logger.handlers.clear()

    # if not logger.handlers:
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger

logger = create_logger(__name__)

def as_json(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        res = f(*args, **kwargs)
        res = json.dumps(res, ensure_ascii=False, default=str).encode('utf-8')
        return Response(res, content_type='application/json; charset=uft-8')
    return decorated_function
###############################################################################
# Set the route and accepted methods

#
# @app.route("/simulator/new", methods=["GET", "POST"])
# @as_json
# def create_simulator():
#     logger.info("create simulator")
#     try:
#         param = request.get_json()
#         print(
#             '----------------------\n'
#             , param
#         )
#         # data = start_url.rest_post(new_url, param, 5, show_error=True)     # param 을 그대로 전달
#         data = start_url.rest_get(new_url, 5, show_error=True)
#         print(data)
#         return data
#         # return services.create_simulator(param)
#     except Exception:
#         result = dict()
#         result["success"] = False
#         result["log"] = traceback.format_exc()
#         logger.error(result["log"])
#         return result
#
#
# @mod_service.route("/simulator/reset", methods=["GET", "POST"])
# @as_json
# def reset_simulator():
#     logger.info("reset simulator")
#     try:
#         param = request.get_json()
#         return services.reset_simulator(param)
#     except Exception:
#         result = dict()
#         result["success"] = False
#         result["log"] = traceback.format_exc()
#         logger.error(result["log"])
#         return result
#
#
# @mod_service.route("/simulator/close", methods=["GET", "POST"])
# @as_json
# def close_simulator():
#     logger.info("close simulator")
#     try:
#         param = request.get_json()
#         print(f"parameter -> {param}")
#         return services.close_simulator(param)
#     except Exception:
#         result = dict()
#         result["success"] = False
#         result["log"] = traceback.format_exc()
#         logger.error(result["log"])
#         return result
#
#
# @mod_service.route("/simulator/get-info", methods=["GET", "POST"])
# @as_json
# def info_simulator():
#     logger.info("info simulator")
#     try:
#         param = request.get_json()
#         return services.info_simulator(param)
#     except Exception:
#         result = dict()
#         result["success"] = False
#         result["log"] = traceback.format_exc()
#         logger.error(result["log"])
#         return result
#
#
# @app.route("/api/v1.0/simulator/check-status", methods=["GET"])
# def status_simulator():
#     logger.info("status simulator")
#     try:
#         param = request.get_json()
#         data = dict()
#         data['result'] = 'success'
#         return data
#     except Exception:
#         result = dict()
#         result["success"] = False
#         result["log"] = traceback.format_exc()
#         print('================\n', result)
#         logger.error(result["log"])
#         return result
#
#
# @mod_service.route("/simulator/step", methods=["GET", "POST"])
# @as_json
# def step_simulator():
#     logger.info("step simulator")
#     try:
#         param = request.get_json()
#         return services.step_simulator(param)
#     except Exception:
#         result = dict()
#         result["success"] = False
#         result["log"] = traceback.format_exc()
#         logger.error(result["log"])
#         return result
#
#
# @mod_service.route("/simulator/set-seed", methods=["GET", "POST"])
# @as_json
# def set_seed_simulator():
#     logger.info("set_seed simulator")
#     try:
#         param = request.get_json()
#         return services.set_seed_simulator(param)
#     except Exception:
#         result = dict()
#         result["success"] = False
#         result["log"] = traceback.format_exc()
#         logger.error(result["log"])
#         return result
#
#
# @mod_service.route("/simulator/argument", methods=["GET", "POST"])
# @as_json
# def test_argument_simulator():
#     logger.info("simulator argument check (reset, step, seed)")
#     param = request.get_json()
#     return services.test_argument_simulator(param)
#
#
# @mod_service.route("/simulator/delete", methods=["GET", "POST"])
# @as_json
# def delete_simulator():
#     logger.info("delete simulator")
#     param = request.get_json()
#     return services.delete_simulator(param)
#
# # @mod_service.route('/simulator/test', methods=['GET', 'POST'])
# # @as_json
# # def test_redis():
# #     logger.info("test remake simulator")
# #     try:
# #         print(services.test_redis())
# #         return "test"
# #     except Exception:
# #         print('test error')
# #         result = dict()
# #         result['success'] = False
# #         result['log'] = traceback.format_exc()
# #         logger.error(result['log'])
# #         return result