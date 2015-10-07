#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2015 Takuma Yagi <takuma.8192.bogin@mbr.nifty.com>
#
# Distributed under terms of the MIT license.

"""
単独単語認識を行う
detection_frame_cntフレーム以上同一の予測が得られるラベルのうち、最も遅く検知されたものを返す
入力：raw_pred list(frame_cnt)
出力：int
"""
def predict_sign_simple(raw_pred, detection_frame_cnt, ignored_labels):
    counter = 1
    candidate = -1
    answer = -1
    for i in range(1, len(raw_pred)):
        if raw_pred[i] in ignored_labels:
            candidate = -1
            counter = 1
            continue
        if raw_pred[i] == candidate:
            counter += 1
            if counter >= detection_frame_cnt:
                answer = candidate
        else:
            candidate = raw_pred[i]
            counter = 1

    return answer

if __name__ == "__main__":
    
    raw_pred = [0, 0, 0, 0, 0, 7, 7, 7, 9, 9, 2, 2, 2, 1, 1, 1, 1, 1] 
    assert predict_sign_simple(raw_pred, 3, [0, 1]) == 2
