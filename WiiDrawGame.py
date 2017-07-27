import sys, random, sched, time, copy
from threading import Thread
import numpy as np
import classifier.quickdraw_npy_bitmap_helper as helper
import classifier.itt_draw_cnn as draw
import wiimote
import wiimote_drawing
import images.images_rc
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

        self.setAttribute(QtCore.Qt.WA_StaticContents)
        self.setMinimumHeight(450)
        self.setMinimumWidth(530)
        self.modified = False
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

        # Undo Test
        self.undoButton = QtWidgets.QPushButton("undo", self)
        self.undoButton.clicked.connect(self.undo)

        # Deactivate during debug
        #self.gameStart = False
    def undo(self):
        self.currentSegmentIndex = max(0, self.currentSegmentIndex -1)
        if len(self.drawing[self.currentSegmentIndex]) == 0:
            self.currentSegmentIndex = max(0, self.currentSegmentIndex - 1)
        self.drawImage()

    def redo(self):
        self.currentSegmentIndex = self.currentSegmentIndex + 1
        self.drawImage()

    def setStartPoint(self, startPoint):
        self.lastPoint = startPoint

    def setPenColor(self, newColor):
        self.myPenColor = newColor

    def setPenWidth(self, newWidth):
        self.myPenWidth = newWidth

    def clearImage(self):
        self.image.fill(QtGui.qRgb(0, 0, 0))
        self.modified = True
        self.drawing = []
        self.drawing.append([])
        self.drawing.append([])
        self.currentSegmentIndex = 1
        self.update()

    def resetCanvas(self):
        self.image.fill(QtGui.qRgb(0, 0, 0))
        self.modified = True
        self.update()

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

    def drawImage(self):
        self.resetCanvas()
        for lineSegment in self.drawing[:self.currentSegmentIndex+1]:
            for line in lineSegment:
                if line:
                    self.lastPoint = line.p1()
                    self.drawLineTo(line.p2())
        self.update()

    def updateDrawing(self, p):
        if self.currentSegmentIndex < len(self.drawing)-1:
            self.currentSegmentIndex = self.currentSegmentIndex + 1
            self.drawing = self.drawing[:self.currentSegmentIndex+1]
            self.drawing[self.currentSegmentIndex] = []
        self.drawLineTo(p)
        self.drawing[self.currentSegmentIndex].append(self.line)

    def drawLineTo(self, endPoint):
        painter = QtGui.QPainter(self.image)
        painter.setPen(QtGui.QPen(self.myPenColor, self.myPenWidth, QtCore.Qt.SolidLine,
                QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        self.line = QtCore.QLine(self.lastPoint, endPoint)
        painter.drawLine(self.line)
        self.modified = True
        rad = self.myPenWidth / 2 + 2
        self.update(QtCore.QRect(self.lastPoint, endPoint).normalized().adjusted(-rad, -rad, +rad, +rad))
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

    def print_(self):
        printer = QtPrintSupport.QPrinter(QtPrintSupport.QPrinter.HighResolution)

        printDialog = QtPrintSupport.QPrintDialog(printer, self)
        if printDialog.exec_() == QtPrintSupport.QPrintDialog.Accepted:
            painter = QtGui.QPainter(printer)
            rect = painter.viewport()
            size = self.image.size()
            size.scale(rect.size(), QtCore.Qt.KeepAspectRatio)
            painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
            painter.setWindow(self.image.rect())
            painter.drawImage(0, 0, self.image)
            painter.end()

    def isModified(self):
        return self.modified

    def penColor(self):
        return self.myPenColor

    def penWidth(self):
        return self.myPenWidth

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
        self.time = 60
        self.winningPoints = 1
        self.currentWord = ""
        self.gameRunning = True
        self.roundWon = False
        self.roundRunning = False
        self.currentTeam = 2
        self.scoreTeamOne = 0
        self.scoreTeamTwo = 0
        self.guess = ""
        self.initUI()
        self.image_clear_count_index = self.time
        self.cw = ScribbleArea(self.ui.frame)
        self.show()
        self.prHelper = helper.QuickDrawHelper()
        self.trainModel = draw.ITTDrawGuesserCNN(self.prHelper.get_num_categories())
        self.trainModel.load_model("classifier/trained_model/draw_game_model_4.tfl")
        self.svm = svm_classifier.SimpleGestureRecognizer()
        self.svm.load_classifier("classifier/svm_model.gz")

        wiimote.buttons.register_callback(self.buttonEvents)
        wiiDraw.register_callback(self.setMousePos)
        wiiDraw.start_processing()

    def initUI(self):
        self.ui.clear.clicked.connect(self.clearImage)
        self.ui.startButton.clicked.connect(self.startGaming)
        self.ui.startGame.clicked.connect(self.startNewRound)
        self.ui.endGame.clicked.connect(self.startNewGame)
        self.ui.timer.display(self.time)
        self.ui.team1Score.display(self.scoreTeamOne)
        self.ui.team2Score.display(self.scoreTeamTwo)
        self.ui.blueTeam.hide()
        backgroundImage = QtGui.QPixmap(':/background/paper.jpg')  # resource path starts with ':'
        redTeamIcon = QtGui.QPixmap(':/teamDots/redDot.png')  # resource path starts with ':'
        blueTeamIcon = QtGui.QPixmap(':/teamDots/blueDot.png')  # resource path starts with ':'
        self.ui.startScreen.setPixmap(backgroundImage)
        self.ui.redTeam.setPixmap(redTeamIcon)
        self.ui.blueTeam.setPixmap(blueTeamIcon)

    def startGaming(self):
        print(self.scoreTeamOne)

        self.ui.title.hide()
        self.ui.startButton.hide()
        self.ui.startScreen.lower()
        self.ui.secondText.hide()
        self.time = int(self.ui.selectSeconds.value())
        self.ui.timer.display(self.time)
        self.ui.selectSeconds.hide()
        self.ui.secondsSlider.hide()


    def startNewGame(self):
        self.setTitleScreen()
        self.time = 60
        self.currentWord = ""
        self.roundWon = False
        self.roundRunning = False
        self.gameRunning = True
        self.currentTeam = 1
        print(self.scoreTeamOne)
        self.scoreTeamOne = 0
        self.scoreTeamTwo = 0
        self.guess = ""
        print(self.scoreTeamOne)
        self.cw.clearImage()
        self.initUI()

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
        self.ui.category.setText("Term")

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
            self.currentWord = "umbrella" #random.choice(words)
            self.guess = ""
            self.ui.timer.display(self.time)
            self.ui.category.setText(self.currentWord)
            self.clearImage()
            self.ui.startGame.setEnabled(False)
            self.ui.kiGuess.setText("I think it is: %s" % self.guess)
            t = Thread(target=self.countdown)
            t.start()

    # Just for testing
    def changeGuess(self, guess):
        self.guess = guess
        self.ui.kiGuess.setText("I think it is: %s" % self.guess)

    def countdown(self):
        x = self.time-1
        for i in range(x, -1, -1):
            if i%3 == 0:
                currentImage = self.cw.saveImage()
                self.changeGuess(self.prHelper.get_label(self.trainModel.predict(currentImage)))
            if i%2 == 0:
                self.cw.addSegment()
            if i % 1 == 0:
                print(i, self.image_clear_count_index - i)
                if self.svm.predict() == 0 and self.image_clear_count_index - i > 2:
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

    # Todo: Just for Testing / Find better solution!
    def processEndRound(self):
        if self.roundWon:
            if self.currentTeam == 1:
                self.scoreTeamOne = self.scoreTeamOne + 1
                self.ui.kiGuess.setText("Oh, i know, the Word is: %s. Team 1 gets 1 Point!" % self.guess)
            else:
                self.scoreTeamTwo = self.scoreTeamTwo + 1
                self.ui.kiGuess.setText("Oh, i know, the Word is: %s. Team 2 get 1 Point!" % self.guess)
        else:
            self.ui.kiGuess.setText("Sorry, i couldn't guess the word!")

        self.ui.team1Score.display(self.scoreTeamOne)
        self.ui.team2Score.display(self.scoreTeamTwo)
        self.checkGameEnd()

    def checkGameEnd(self):
        if self.scoreTeamOne == self.winningPoints:
            self.ui.kiGuess.setText("Team 1 has %s Points. Team 1 won!" % self.winningPoints)
            self.gameRunning = False
            self.ui.startGame.setEnabled(False)
        elif self.scoreTeamTwo == self.winningPoints:
            self.ui.kiGuess.setText("Team 2 has %s Points. Team 2 won!" % self.winningPoints)
            self.gameRunning = False
            self.ui.startGame.setEnabled(False)
        self.roundRunning = False
        self.ui.startGame.setEnabled(True)
        self.cw.gameRuns = False

    def checkGuessing(self):
        if self.guess == self.currentWord:
            self.roundWon = True

    def clearImage(self):
        self.cw.clearImage()

    def saveFile(self, fileFormat):
        return self.cw.saveImage("test", fileFormat)
        return False

    def setNewColor(self):
        col = QtWidgets.QColorDialog.getColor()
        if col.isValid():
            self.cw.setPenColor(col)

    def setMousePos(self, pos, acc):

        x, y, z = acc
        self.svm.update_buffer(x, y, z)

        if pos == None:
            return
        QtGui.QCursor.setPos(self.mapToGlobal(QtCore.QPoint(pos[0], pos[1])))

    def buttonEvents(self, report):
        for button in report:
            if button[0] == "A" and not button[1]:
                self.cw.undo()
            if button[0] == "B" and button[1]:
                pyautogui.mouseDown(button="left")
            elif button[0] == "B" and not button[1]:
                pyautogui.mouseUp(button="left")


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
    wiimote = connect_wiimote()
    wiiDraw = wiimote_drawing.init(wiimote)
    paint = Painter(wiimote, wiiDraw)
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
