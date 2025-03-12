import cv2
import mediapipe as mp
import numpy as np
import pygame
import random

class handDetector():

    def __init__(self, mode = False, maxHands =2, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = int(detectionCon)
        self.trackCon = int(trackCon)
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

        self.finger_tips =[4, 8, 12, 16, 20]

    def findHands(self, frame, draw = True):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.result = self.hands.process(rgb_frame)

        # Detect hand and draw landmarks
        if self.result.multi_hand_landmarks:
            for hand_landmarks in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, hand_landmarks, 
                                           self.mpHands.HAND_CONNECTIONS)
        return frame

    def findPosition(self, frame, handNo = 0, draw = False):
        self.landmarks = []

        if self.result.multi_hand_landmarks:
            # myHand = self.result.multi_hand_landmarks[handNo]

            for myHand in self.result.multi_hand_landmarks:
                hand = []
                for id, landmark in enumerate(myHand.landmark):
                    height, width, channel = frame.shape
                    cx, cy = int(landmark.x*width), int(landmark.y*height)
                    hand.append([id,cx,cy])
                self.landmarks.append(hand)

                if draw:
                    for i  in hand:
                        cv2.circle(frame, (i[1],i[2]), 4, (0,255,255), cv2.FILLED)
               
        return self.landmarks
    
    def fingerUp(self):
        fingersup = []
        if self.landmarks == []:
            return None, None
        
        for hand in self.landmarks:
            fingers = []
            
            # Check if hand is left or right
            is_left = hand[17][1] < hand[9][1]

            # Thumb detection
            if is_left:
                if hand[self.finger_tips[0]][1] > hand[self.finger_tips[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                if hand[self.finger_tips[0]][1] < hand[self.finger_tips[0] - 1][1]:   # Reverse condition
                    fingers.append(1)
                else:
                    fingers.append(0)

            # Finger detection
            for id in range(1, 5):
                if hand[self.finger_tips[id]][2] < hand[self.finger_tips[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            fingersup.append(fingers)

        if len(fingersup) == 1:
            return fingersup[0], None
        return fingersup[0], fingersup[1]



def main():
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    prev_time = curr_time = 0

    while True:
        success, frame = cap.read()
        frame = detector.findHands(frame)
        landmarks_list =detector.findPosition(frame)
        # if landmarks_list != []:
        #     fingers =detector.fingerUp()
        
        cv2.imshow("Hand", frame)
        cv2.waitKey(1)

        # curr_time - time.time()
        # fps = 1 / (curr_time - prev_time)
        # prev_time = curr_time


if __name__ == "__main__":
    main()

