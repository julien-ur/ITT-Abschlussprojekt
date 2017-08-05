import sys, random, sched, time, copy
from threading import Thread
import numpy as np
import classifier.quickdraw_npy_bitmap_helper as helper
import classifier.itt_draw_cnn as draw
import wiimote
import wiimote_drawing
import images.images
import pyautogui
import classifier.svm_gesture_classifier as svm_classifier

from PyQt5 import uic, QtWidgets, QtCore, QtGui, QtPrintSupport, Qt, QtTest
from PyQt5.QtGui import qRgb

import qimage2ndarray

p1 = QtCore.QPoint(0, 0)
p2 = QtCore.QPoint(400, 400)

words = []
# Reduced Categories for Testing
words = [line.rstrip('\n') for line in open('categories.txt')]

# Source: https://github.com/baoboa/pyqt5/blob/master/examples/widgets/scribble.py
# Used the scribble.py example as basis for the drawing area. The saveImage function was overwritten with the creation
# of the qimage2ndarray. For this, we are using the python extension: https://github.com/hmeine/qimage2ndarray

class ScribbleArea(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(ScribbleArea, self).__init__(parent)
        self.scribbling = False
        self.myPenWidth = 3
        self.myPenColor = QtCore.Qt.white
        self.image = QtGui.QImage()
        self.lastPoint = QtCore.QPoint()

        self.gameRuns = False

        self.drawingSegment = []
        self.drawing = []
        self.drawing.append([])
        self.drawing.append([])
        self.currentSegmentIndex = 1

    def undo(self):
        self.currentSegmentIndex = max(0, self.currentSegmentIndex - 1)

        if len(self.drawing[self.currentSegmentIndex]) == 0:
            self.currentSegmentIndex = max(0, self.currentSegmentIndex - 1)

        self.drawImage()

    def resizeCanvas(self, height, width):
        self.setMinimumHeight(height)
        self.setMinimumWidth(width)
        self.update()

    def redo(self):
        self.currentSegmentIndex = self.currentSegmentIndex + 1
        self.drawImage()

    def setStartPoint(self, startPoint):
        self.lastPoint = startPoint

    def clearImage(self):
        self.image.fill(QtGui.qRgb(0, 0, 0))
        self.drawing = []
        self.drawing.append([])
        self.drawing.append([])
        self.currentSegmentIndex = 1
        self.update()

    def resetCanvas(self):
        self.image.fill(QtGui.qRgb(0, 0, 0))
        self.update()

    # Save current canvas to an ndarray (https://github.com/hmeine/qimage2ndarray)
    def saveImage(self):
        v = qimage2ndarray.rgb_view(self.image)
        return v

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.lastPoint = event.pos()
            self.scribbling = True

    def mouseMoveEvent(self, event):
        if (event.buttons() & QtCore.Qt.LeftButton) and self.scribbling:
            if self.gameRuns:
                self.updateDrawing(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.scribbling:
            if self.gameRuns:
                self.updateDrawing(event.pos())
                self.scribbling = False

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        dirtyRect = event.rect()
        painter.drawImage(dirtyRect, self.image, dirtyRect)

    def resizeEvent(self, event):
        if self.width() > self.image.width() or self.height() > self.image.height():
            newWidth = max(self.width() + 128, self.image.width())
            newHeight = max(self.height() + 128, self.image.height())
            self.resizeImage(self.image, QtCore.QSize(newWidth, newHeight))
            self.update()

        super(ScribbleArea, self).resizeEvent(event)

    # Called when undo function is executed / Draw new Image based in segment index
    def drawImage(self):
        self.resetCanvas()
        for lineSegment in self.drawing[:self.currentSegmentIndex+1]:
            for line in lineSegment:
                if line:
                    self.lastPoint = line.p1()
                    self.drawLineTo(line.p2())
        self.update()

    # Called when mouse moves and the press is released / Draw new lines
    def updateDrawing(self, p):
        if self.currentSegmentIndex < len(self.drawing)-1:
            self.currentSegmentIndex = self.currentSegmentIndex + 1
            self.drawing = self.drawing[:self.currentSegmentIndex+1]
            self.drawing[self.currentSegmentIndex] = []
        self.drawLineTo(p)
        self.update()
        self.drawing[self.currentSegmentIndex].append(self.line)

    # Draws line between to points
    def drawLineTo(self, endPoint):
        painter = QtGui.QPainter(self.image)
        painter.setPen(QtGui.QPen(self.myPenColor, self.myPenWidth, QtCore.Qt.SolidLine,
                QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        self.line = QtCore.QLine(self.lastPoint, endPoint)
        painter.drawLine(self.line)
        rad = self.myPenWidth / 2 + 2
        #self.update(QtCore.QRect(self.lastPoint, endPoint).normalized().adjusted(-rad, -rad, +rad, +rad))
        self.lastPoint = QtCore.QPoint(endPoint)

    def resizeImage(self, image, newSize):
        if image.size() == newSize:
            return

        newImage = QtGui.QImage(newSize, QtGui.QImage.Format_RGB32)
        newImage.fill(qRgb(0, 0, 0))
        painter = QtGui.QPainter(newImage)
        painter.drawImage(QtCore.QPoint(0, 0), image)
        self.image = newImage
        self.update()

    # Add new segment to undo stack
    def addSegment(self):
        if self.currentSegmentIndex < len(self.drawing)-1:
            return
        if len(self.drawing[self.currentSegmentIndex]) == 0:
            return
        self.drawing.append([])
        self.currentSegmentIndex = len(self.drawing)-1

class Painter(QtWidgets.QMainWindow):
    def __init__(self, wiimote, wiiDraw):
        super(Painter, self).__init__()
        self.ui = uic.loadUi("DrawGame.ui", self)
        self.time = 30
        self.winningPoints = 2
        self.currentWord = ""
        self.gameRunning = True
        self.roundWon = False
        self.roundRunning = False
        self.currentTeam = 2
        self.scoreTeamOne = 0
        self.scoreTeamTwo = 0
        self.guess = ""
        pyautogui.FAILSAFE = False
        self.initUI()
        self.image_clear_count_index = self.time
        self.image_undo_count_index = self.time
        self.cw = ScribbleArea(self.ui.frame)
        self.show()
        self.prHelper = helper.QuickDrawHelper()
        self.trainModel = draw.ITTDrawGuesserCNN(self.prHelper.get_num_categories())
        self.trainModel.load_model("classifier/trained_model/draw_game_model_4.tfl")
        self.svm = svm_classifier.SimpleGestureRecognizer()
        self.svm.load_classifier("classifier/svm_model.gz")
        self.showFullScreen()
        screen = QtWidgets.QDesktopWidget().screenGeometry()
        self.backgroundSize = self.ui.startScreen.size()
        self.scaleFactorWidth = screen.width()/self.backgroundSize.width()
        self.scaleFactorHeight = screen.height()/self.backgroundSize.height()
        self.cw.resizeCanvas(self.ui.frame.height()*self.scaleFactorHeight, self.ui.frame.width()*self.scaleFactorWidth)
        self.uiElements = self.ui.children()

        for el in self.uiElements:
            if isinstance(el, QtWidgets.QLayout):
                continue
            else:
                el.setGeometry(QtCore.QRect(el.x()*self.scaleFactorWidth,el.y()*self.scaleFactorHeight, el.width()*self.scaleFactorWidth, el.height()*self.scaleFactorHeight ))

        #wiimote.buttons.register_callback(self.buttonEvents)
        #wiiDraw.register_callback(self.setMousePos)
        #wiiDraw.start_processing()

    # Initalize UI Elements
    def initUI(self):
        self.ui.clear.clicked.connect(self.clearImage)
        self.ui.startButton.clicked.connect(self.startGaming)
        self.ui.startGame.clicked.connect(self.startNewRound)
        self.ui.endGame.clicked.connect(self.startNewGame)
        self.ui.timer.display(self.time)
        self.ui.team1Score.display(self.scoreTeamOne)
        self.ui.team2Score.display(self.scoreTeamTwo)
        self.ui.blueTeam.hide()
        backgroundImage = QtGui.QPixmap(':/background/table.jpg')  # resource path starts with ':'
        redTeamIcon = QtGui.QPixmap(':/teams/redDot.png')  # resource path starts with ':'
        blueTeamIcon = QtGui.QPixmap(':/teams/blueDot.png')  # resource path starts with ':'
        self.ui.startScreen.setPixmap(backgroundImage)
        self.ui.redTeam.setPixmap(redTeamIcon)
        self.ui.blueTeam.setPixmap(blueTeamIcon)

    # Hide title screen elements
    def startGaming(self):
        self.ui.title.hide()
        self.ui.startButton.hide()
        self.ui.startScreen.lower()
        self.ui.secondText.hide()
        self.time = int(self.ui.selectSeconds.value())
        self.ui.timer.display(self.time)
        self.ui.selectSeconds.hide()
        self.ui.secondsSlider.hide()

    # Called when current game aborts or gets won/ all vars get set to start value
    def startNewGame(self):
        self.setTitleScreen()
        self.time = 60
        self.currentWord = "Term"
        self.roundWon = False
        self.roundRunning = False
        self.gameRunning = True
        self.currentTeam = 1
        self.scoreTeamOne = 0
        self.scoreTeamTwo = 0
        self.guess = ""
        self.cw.clearImage()
        self.initUI()

    # Used to bring all title screen elements to the top
    def setTitleScreen(self):
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
    def startNewRound(self):
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
            self.clearImage()
            self.ui.startGame.setEnabled(False)
            self.ui.kiGuess.setText("I think it is: %s" % self.guess)
            t = Thread(target=self.countdown)
            t.start()

    # Set new guess
    def changeGuess(self, guess):
        self.guess = guess
        self.ui.kiGuess.setText("I think it is: %s" % self.guess)

    # Handles what happens in the Countdown Thread
    def countdown(self):
        x = self.time-1
        for i in range(x, -1, -1):
            if i%3 == 0:
                currentImage = self.cw.saveImage()
                self.changeGuess(self.prHelper.get_label(self.trainModel.predict(currentImage)))
            if i%2 == 0:
                self.cw.addSegment()
            if i%1 == 0:
                gesture = self.svm.predict()
                if gesture == 1 and abs(self.image_clear_count_index - i) >= 1:
                    self.image_clear_count_index = i
                    self.clearImage()

            if not self.roundWon:
                time.sleep(1)
                self.ui.timer.display(i)
                self.checkGuessing()
            else:
                break
        self.processEndRound()
        print("Round End")

    # Check if team has won point
    def processEndRound(self):
        if self.roundWon:
            if self.currentTeam == 1:
                self.scoreTeamOne = self.scoreTeamOne + 1
            else:
                self.scoreTeamTwo = self.scoreTeamTwo + 1
            self.ui.kiGuess.setText("Oh i know, the Word is: %s. Team %s gets 1 Point!" % (self.guess, self.currentTeam))

        else:
            self.ui.kiGuess.setText("Sorry, i couldn't guess the word!")

        self.ui.team1Score.display(self.scoreTeamOne)
        self.ui.team2Score.display(self.scoreTeamTwo)
        self.checkGameEnd()

    # Check if game is won
    def checkGameEnd(self):
        if (self.scoreTeamOne == self.winningPoints) or (self.scoreTeamTwo == self.winningPoints):
            self.ui.kiGuess.setText(
                "Team %s has %s Points. Team %s won!" % (self.currentTeam, self.winningPoints, self.currentTeam))
            self.gameRunning = False
            self.ui.startGame.setEnabled(False)
        self.roundRunning = False
        self.ui.startGame.setEnabled(True)
        self.cw.gameRuns = False

    # Checks if KI guessed the word
    def checkGuessing(self):
        if self.guess == self.currentWord:
            self.roundWon = True

    # Delete whole image
    def clearImage(self):
        self.cw.clearImage()

    # Set new cursor pos to a new pos
    def setMousePos(self, pos, acc):
        start = time.time()
        x, y, z = acc
        self.svm.update_buffer(x, y, z)

        if pos == None:
            return
        QtGui.QCursor.setPos(self.mapToGlobal(QtCore.QPoint(pos[0], pos[1])))
        end = time.time()
        print(end - start)

    # Button events for the Wiimote. Used PyAutoGui for interaction with app (link to pyautogui)
    def buttonEvents(self, report):
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
    #wiimote = connect_wiimote()
    #wiiDraw = wiimote_drawing.init(wiimote)
    paint = Painter(None, None)
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
