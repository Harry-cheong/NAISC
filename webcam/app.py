# import the opencv library
import cv2
# define a video capture object
vid = cv2.VideoCapture(0)

keep_running = True
generate_compliment = True
while(keep_running):
    while(keep_running):
        ret, frame = vid.read()
        cv2.imshow('frame', frame)
        k = cv2.waitKey(1)
        if k == ord('q'):
            keep_running = False
            break
        if k == ord('c'):
            generate_compliment = True
            break
        if k == ord('i'):
            generate_compliment = False
            break

    while(keep_running):
        cv2.imshow('frame', frame)
        # find some way to let them adjust attitude value
        display_text = "Generating compliment..." if generate_compliment else "Generating insult..."
        cv2.putText(frame, display_text, (50,50), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), thickness=2)
        k = cv2.waitKey(1)
        if k == ord('q'):
            keep_running = False
            break
        if k == ord(' '):
            # replace this with a when insult/compliement ready
            break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()