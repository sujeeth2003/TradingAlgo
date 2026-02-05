"""
Market Data Module
==================
Fetches price data from Polygon.io (free tier) or Yahoo Finance as fallback.
Falls back to synthetic GBM data if no API key is set.
"""

import os
import json
import math
import random
import urllib.request
import urllib.parse
from datetime import datetime, timedelta

