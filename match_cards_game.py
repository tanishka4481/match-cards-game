import cv2
import numpy as np
import random
from sklearn.utils import shuffle
import hand_draw as hd
import mediapipe as md
import time


""" ##################################################################### """
""" ##################################################################### """
""" ##################################################################### """
""" ##################################################################### """
""" ##################################################################### """

# Load images for fornt faces of cards
img_path = [f"C:/Users/Hp/OneDrive/Documents/Btech_first/Practice_home/openCV/project/animal_pictures/{i}.png" for i in range(13,21)]
imgs_front_face = [cv2.imread(img) for img in img_path]
imgs_front_face = [cv2.resize(img, (140, 140)) for img in imgs_front_face]

imgs_front_face = imgs_front_face * 2
# Shuffle images
random.shuffle(imgs_front_face)

img_path = [f"C:/Users/Hp/OneDrive/Documents/Btech_first/Practice_home/openCV/project/animal_pictures/{i}.png" for i in range(1,13)]
imgs_back_face = [cv2.imread(img) for img in img_path]
imgs_back_face = [cv2.resize(img, (140, 140)) for img in imgs_back_face]

# Select 8 random images
selected_img_back = random.sample(imgs_back_face, 8)
imgs_back_face = selected_img_back * 2

# Assign indices and shuffle each pair (index-image)
indices = [i for i in range(1, 9)] * 2
indices, imgs_back_face = shuffle(indices, imgs_back_face, random_state=0)

# Store all images to display
image_show = {i+1: [indices[i], imgs_front_face[i], imgs_back_face[i]] for i in range(16)}

# Create a blank canvas
blank = np.zeros((800, 800, 3), dtype='uint8')

selected_cards = []
matched_cards = set()

def display_cards(overlay):
    """ Update the frame to display cards. """
    i = 1  
    for row in range(4):
        y_start = row * 150 + 20
        for col in range(4):
            x_start = col * 150 + 20
            row_col_text = f"{row},{col}"

            if i in matched_cards:
                img = np.zeros((140, 140, 3), dtype=np.uint8)  # Remove matched card (Black square)
            elif i in selected_cards:
                img = image_show[i][2]  # Show back
            else:
                img = image_show[i][1]  # Show front
            
            # create border around selected image
            # img_with_frame = cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])

            overlay[y_start:y_start + 140, x_start:x_start + 140] = img

            # Display card positions
            cv2.putText(overlay, row_col_text, (x_start + 10, y_start + 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
            i += 1

    return overlay


gesture_buffer = []
CONFIRM_COUNT = 12  

def get_stable_selection(fingers_1, fingers_2):
    # Returns a stable selection only if the same gesture is repeated 12 times in a row.
    current_selection = sum(fingers_1) * 4 + sum(fingers_2) + 1

    # Add the latest selection to the buffer
    gesture_buffer.append(current_selection)

    # Keep only the last 12 entries
    if len(gesture_buffer) > CONFIRM_COUNT:
        gesture_buffer.pop(0)

    # Confirm selection if all 12 recent entries are the same
    if len(gesture_buffer) == CONFIRM_COUNT and len(set(gesture_buffer)) == 1:
        return current_selection  

    return None  # No stable selection yet


def match_card():
    # Check if two selected cards match and update the game state.
    global selected_cards, matched_cards

    if len(selected_cards) == 2:
        val1, val2 = selected_cards
        print(f"Checking Match: {val1} & {val2}")

        if not (1 <= val1 <= 16) or not (1 <= val2 <= 16):
            print("Invalid selection! Ignoring...")
            selected_cards = []  # Reset selection
            return False

        if np.array_equal(image_show[val1][0], image_show[val2][0]):
            print("Matched!")
            matched_cards.add(val1)
            matched_cards.add(val2)
        else:
            print("Not a match.")
            time.sleep(1)  # Short delay before flipping back

        selected_cards = []

        # Check if all cards are matched - won or not
        if len(matched_cards) == 16:
            return True 
    return False 


cap = cv2.VideoCapture(0)
detector = hd.handDetector()

while True:
    success, frame = cap.read()
    frame = detector.findHands(frame)
    landmarks_list = detector.findPosition(frame)
    frame = cv2.resize(frame, (1200, 750))
    frame = cv2.flip(frame, 1)

    fingers_1, fingers_2 = detector.fingerUp()

    if fingers_1 is not None and fingers_2 is not None:
        remove_card = get_stable_selection(fingers_1, fingers_2)

        if remove_card is not None and remove_card not in matched_cards and remove_card not in selected_cards:
            selected_cards.append(remove_card)

    overlay = display_cards(frame)
    combined = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    cv2.imshow("Memory Game", combined)

    if len(selected_cards) == 2:
        if match_card():  # Check for game completion
            print("YOU WON!")
            cv2.putText(frame, "YOU WON!", (400, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
            cv2.imshow("Memory Game", frame)
            cv2.waitKey(3000) 
            break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()