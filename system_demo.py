import sys
import random
import time
from threading import Thread
import classifier.quickdraw_npy_bitmap_helper as helper
import classifier.itt_draw_cnn as draw
import wiimote
import wiimote_drawing
import images.images
import pyautogui
import classifier.svm_gesture_classifier as svm_classifier
from PyQt5 import uic, QtWidgets, QtCore, QtGui
from PyQt5.QtGui import qRgb
import qimage2ndarray


words = [line.rstrip('\n') for line in open('categories.txt')]

# Source: https://github.com/baoboa/pyqt5/blob/master/examples/widgets/scribble.py
# Used the scribble.py example as basis for the drawing area. The saveImage function was overwritten with the creation
# of the qimage2ndarray. For this, we are using the python extension: https://github.com/hmeine/qimage2ndarray


class ScribbleArea(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(ScribbleArea, self).__init__(parent)
        self.scribbling = False
        self.penWidth = 3
        self.penColor = QtCore.Qt.white
        self.image = QtGui.QImage()
        self.lastPoint = QtCore.QPoint()

        self.gameRuns = False

        self.drawingSegment = []
        self.drawing = []
        self.drawing.append([])
        self.drawing.append([])
        self.currentSegmentIndex = 1

    # undo function: segmentIndex gets reduced by one
    def undo(self):
        self.currentSegmentIndex = max(0, self.currentSegmentIndex - 1)

        if len(self.drawing[self.currentSegmentIndex]) == 0:
            self.currentSegmentIndex = max(0, self.currentSegmentIndex - 1)

        self.draw_image()

    def resize_canvas(self, height, width):
        self.setMinimumHeight(height)
        self.setMinimumWidth(width)
        self.update()

    def redo(self):
        self.currentSegmentIndex = self.currentSegmentIndex + 1
        self.draw_image()

    def set_start_point(self, start_point):
        self.lastPoint = start_point

    def clear_image(self):
        self.drawing = []
        self.drawing.append([])
        self.drawing.append([])
        self.currentSegmentIndex = 1
        self.reset_canvas()

    def reset_canvas(self):
        self.image.fill(QtGui.qRgb(0, 0, 0))
        self.update()

    # Save current canvas to an ndarray (https://github.com/hmeine/qimage2ndarray)
    def save_image(self):
        v = qimage2ndarray.rgb_view(self.image)
        return v

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.lastPoint = event.pos()
            self.scribbling = True

    def mouseMoveEvent(self, event):
        if (event.buttons() & QtCore.Qt.LeftButton) and self.scribbling:
            if self.gameRuns:
                self.update_drawing(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.scribbling:
            if self.gameRuns:
                self.update_drawing(event.pos())
                self.scribbling = False

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        paint_rect = event.rect()
        painter.drawImage(paint_rect, self.image, paint_rect)

    def resizeEvent(self, event):
        if self.width() > self.image.width() or self.height() > self.image.height():
            new_width = max(self.width() + 128, self.image.width())
            new_height = max(self.height() + 128, self.image.height())
            self.resize_image(self.image, QtCore.QSize(new_width, new_height))
            self.update()

        super(ScribbleArea, self).resizeEvent(event)

    # Called when undo function is executed / Draw new Image based on segment index
    def draw_image(self):
        self.reset_canvas()
        for lineSegment in self.drawing[:self.currentSegmentIndex + 1]:
            for line in lineSegment:
                if line:
                    self.lastPoint = line.p1()
                    self.draw_line_to(line.p2())
        self.update()

    # Called when mouse moves and the press is released / Draw new lines
    def update_drawing(self, p):
        if self.currentSegmentIndex < len(self.drawing) - 1:
            self.currentSegmentIndex = self.currentSegmentIndex + 1
            self.drawing = self.drawing[:self.currentSegmentIndex + 1]
            self.drawing[self.currentSegmentIndex] = []
        self.draw_line_to(p)
        self.update()
        self.drawing[self.currentSegmentIndex].append(self.line)

    # Draws line between to points
    def draw_line_to(self, end_point):
        painter = QtGui.QPainter(self.image)
        painter.setPen(QtGui.QPen(self.penColor, self.penWidth, QtCore.Qt.SolidLine,
                                  QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        self.line = QtCore.QLine(self.lastPoint, end_point)
        painter.drawLine(self.line)
        self.lastPoint = QtCore.QPoint(end_point)

    def resize_image(self, image, new_size):
        if image.size() == new_size:
            return

        new_image = QtGui.QImage(new_size, QtGui.QImage.Format_RGB32)
        new_image.fill(qRgb(0, 0, 0))
        painter = QtGui.QPainter(new_image)
        painter.drawImage(QtCore.QPoint(0, 0), image)
        self.image = new_image
        self.update()

    # Add new segment to undo list
    def add_segment(self):
        if self.currentSegmentIndex < len(self.drawing) - 1:
            return
        if len(self.drawing[self.currentSegmentIndex]) == 0:
            return
        self.drawing.append([])
        self.currentSegmentIndex = len(self.drawing) - 1


class Painter(QtWidgets.QMainWindow):
    def __init__(self, wiimote, wiidraw):
        super(Painter, self).__init__()
        self.ui = uic.loadUi("DrawGame.ui", self)
        self.time = 30
        self.winningPoints = 3
        self.currentWord = ""
        self.gameRunning = True
        self.roundWon = False
        self.roundRunning = False
        self.currentTeam = 2
        self.scoreTeamOne = 0
        self.scoreTeamTwo = 0
        self.guess = ""
        pyautogui.FAILSAFE = False
        self.cw = ScribbleArea(self.ui.frame)
        self.init_ui()
        self.image_clear_count_index = self.time
        self.image_undo_count_index = self.time
        self.show()
        self.prHelper = helper.QuickDrawHelper()
        self.trainModel = draw.ITTDrawGuesserCNN(self.prHelper.get_num_categories())
        self.trainModel.load_model("classifier/trained_model/draw_game_model_4.tfl")
        self.svm = svm_classifier.SimpleGestureRecognizer()
        self.svm.load_classifier("classifier/svm_model.gz")
        self.showFullScreen()
        screen = QtWidgets.QDesktopWidget().screenGeometry()
        self.backgroundSize = self.ui.startScreen.size()
        self.scaleFactorWidth = screen.width() / self.backgroundSize.width()
        self.scaleFactorHeight = screen.height() / self.backgroundSize.height()
        self.cw.resize_canvas(self.ui.frame.height() * self.scaleFactorHeight,
                              self.ui.frame.width() * self.scaleFactorWidth)
        self.uiElements = self.ui.children()

        for el in self.uiElements:
            if isinstance(el, QtWidgets.QLayout):
                continue
            else:
                el.setGeometry(QtCore.QRect(el.x() * self.scaleFactorWidth, el.y() * self.scaleFactorHeight,
                                            el.width() * self.scaleFactorWidth, el.height() * self.scaleFactorHeight))

        try:
            wiimote.buttons.register_callback(self.buttonEvents)
            wiidraw.register_callback(self.setMousePos)
            wiidraw.start_processing()
        except:
            print(sys.exc_info()[0])

    # Initalize UI Elements
    def init_ui(self):
        self.ui.undo.clicked.connect(self.cw.undo)
        self.ui.startButton.clicked.connect(self.start_gaming)
        self.ui.startGame.clicked.connect(self.start_new_round)
        self.ui.endGame.clicked.connect(self.start_new_game)
        self.ui.timer.display(self.time)
        self.ui.team1Score.display(self.scoreTeamOne)
        self.ui.team2Score.display(self.scoreTeamTwo)
        self.ui.blueTeam.hide()
        background_image = QtGui.QPixmap(':/background/table.jpg')  # resource path starts with ':'
        red_team_icon = QtGui.QPixmap(':/teams/redDot.png')  # resource path starts with ':'
        blue_team_icon = QtGui.QPixmap(':/teams/blueDot.png')  # resource path starts with ':'
        self.ui.startScreen.setPixmap(background_image)
        self.ui.redTeam.setPixmap(red_team_icon)
        self.ui.blueTeam.setPixmap(blue_team_icon)

    # Hide title screen elements
    def start_gaming(self):
        self.ui.title.hide()
        self.ui.startButton.hide()
        self.ui.startScreen.lower()
        self.ui.secondText.hide()
        self.time = int(self.ui.selectSeconds.value())
        self.ui.timer.display(self.time)
        self.ui.selectSeconds.hide()
        self.ui.secondsSlider.hide()

    # Called when current game aborts or gets won/ all vars get set to start value
    def start_new_game(self):
        self.roundWon = True
        self.ui.startGame.setEnabled(True)
        self.set_title_screen()
        self.time = 60
        self.currentWord = ""
        self.scoreTeamOne = 0
        self.scoreTeamTwo = 0
        self.ui.team1Score.display(self.scoreTeamOne)
        self.ui.team2Score.display(self.scoreTeamTwo)
        self.ui.category.setText(self.currentWord)
        self.guess = ""
        self.ui.kiGuess.setText("I think it is: %s" % self.guess)
        self.roundWon = False
        self.roundRunning = False
        self.gameRunning = True
        self.currentTeam = 1
        self.cw.clear_image()
        self.ui.blueTeam.hide()

    # Used to bring all title screen elements to the top
    def set_title_screen(self):
        self.ui.startScreen.raise_()
        self.ui.title.show()
        self.ui.title.raise_()
        self.ui.startButton.show()
        self.ui.startButton.raise_()
        self.ui.secondText.show()
        self.ui.secondText.raise_()
        self.ui.selectSeconds.show()
        self.ui.selectSeconds.raise_()
        self.ui.secondsSlider.show()
        self.ui.secondsSlider.raise_()
        self.ui.kiGuess.setText("")
        self.ui.category.setText(self.currentWord)

    # Is called when users presses New Round / Starts a new round
    def start_new_round(self):
        if self.gameRunning:
            # Change icon above Team
            if self.currentTeam == 1:
                self.currentTeam = 2
                self.ui.redTeam.hide()
                self.ui.blueTeam.show()
            else:
                self.currentTeam = 1
                self.ui.redTeam.show()
                self.ui.blueTeam.hide()

            self.cw.gameRuns = True
            self.roundWon = False
            self.roundRunning = True
            self.currentWord = random.choice(words)
            self.guess = ""
            self.ui.timer.display(self.time)
            self.ui.category.setText(self.currentWord)
            self.clear_image()
            self.ui.startGame.setEnabled(False)
            self.ui.kiGuess.setText("I think it is: %s" % self.guess)
            self.t = Thread(target=self.countdown)
            self.t.start()

    # Set new guess
    def change_guess(self, guess):
        self.guess = guess
        self.ui.kiGuess.setText("I think it is: %s" % self.guess)

    # Handles what happens in the Countdown Thread
    def countdown(self):
        x = self.time - 1
        for i in range(x, -1, -1):
            if i % 3 == 0:
                current_image = self.cw.save_image()
                self.change_guess(self.prHelper.get_label(self.trainModel.predict(current_image)))

            if i % 2 == 0:
                self.cw.add_segment()

            if i % 1 == 0:
                gesture = self.svm.predict()
                if gesture == 1 and abs(self.image_clear_count_index - i) >= 1:
                    self.image_clear_count_index = i
                    self.clear_image()

            if not self.roundWon:
                time.sleep(1)
                self.ui.timer.display(i)
                self.check_guessing()

            else:
                break
        self.process_end_round()

    # Check if team has won point
    def process_end_round(self):
        if self.roundWon:
            if self.currentTeam == 1:
                self.scoreTeamOne = self.scoreTeamOne + 1
            else:
                self.scoreTeamTwo = self.scoreTeamTwo + 1
            self.ui.kiGuess.setText(
                "Oh i know, the Word is: %s. Team %s gets 1 Point!" % (self.guess, self.currentTeam))
        else:
            self.ui.kiGuess.setText("Sorry, i couldn't guess the word!")

        self.ui.team1Score.display(self.scoreTeamOne)
        self.ui.team2Score.display(self.scoreTeamTwo)
        self.check_game_end()

    # Check if game is won
    def check_game_end(self):
        if (self.scoreTeamOne == self.winningPoints) or (self.scoreTeamTwo == self.winningPoints):
            self.ui.kiGuess.setText(
                "Team %s has %s Points. Team %s won!" % (self.currentTeam, self.winningPoints, self.currentTeam))
            self.gameRunning = False
            self.ui.startGame.setEnabled(False)
            self.roundRunning = False
            self.cw.gameRuns = False
            self.roundWon = True

        else:
            self.roundRunning = False
            self.ui.startGame.setEnabled(True)
            self.cw.gameRuns = False

    # Checks if KI guessed the word
    def check_guessing(self):
        if self.guess == self.currentWord:
            self.roundWon = True

    # Delete whole image
    def clear_image(self):
        self.cw.clear_image()

    # Set new cursor pos to a new pos
    def set_mouse_pos(self, pos, acc):
        x, y, z = acc
        self.svm.update_buffer(x, y, z)

        if pos is None:
            return
        QtGui.QCursor.setPos(self.mapToGlobal(QtCore.QPoint(pos[0], pos[1])))

    # Button events for the Wiimote. Used PyAutoGui for interaction with app
    #  (https://pyautogui.readthedocs.io/en/latest/)
    def button_events(self, report):
        for button in report:
            if button[0] == "A" and not button[1]:
                self.cw.undo()
            if button[0] == "B" and button[1]:
                pyautogui.mouseDown(button="left")
            elif button[0] == "B" and not button[1]:
                pyautogui.mouseUp(button="left")


# Establish connection to Wiimote
def connect_wiimote(btaddr="18:2a:7b:f4:bc:65", attempt=0):
    if len(btaddr) == 17:
        print("connecting wiimote " + btaddr + "..")
        w = None
        try:
            w = wiimote.connect(btaddr)
        except:
            print(sys.exc_info())
            pass
        if w is None:
            print("couldn't connect wiimote. tried it " + str(attempt) + " times")
            time.sleep(3)
            return connect_wiimote(btaddr, attempt + 1)
        else:
            print("succesfully connected wiimote")
            return w
    else:
        print("bluetooth address has to be 17 characters long")
        return None


def main():
    app = QtWidgets.QApplication(sys.argv)
    wiimote, wiidraw = None, None

    if len(sys.argv) > 1:
        wiimote = connect_wiimote(sys.argv[1])
        wiidraw = wiimote_drawing.init(wiimote)

    paint = Painter(wiimote, wiidraw)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
