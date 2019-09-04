#!/usr/bin/env bash
wrk2 -t16 -c64 -d60s -R10000 --latency -s post.lua "http://0.0.0.0:8887/classify_image_vector"