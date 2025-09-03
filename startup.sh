#!/bin/bash
uvicorn app:combined_app --host 0.0.0.0 --port 8000 --workers 8
