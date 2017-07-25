import sys, random, sched, time, copy
from threading import Thread
import numpy as np
import classifier.quickdraw_npy_bitmap_helper as helper
import classifier.itt_draw_cnn as draw
import wiimote
import wiimote_drawing

from PyQt5 import uic, QtWidgets, QtCore, QtGui, QtPrintSupport, Qt, QtTest
from PyQt5.QtGui import qRgb

import qimage2ndarray

p1 = QtCore.QPoint(0, 0)
p2 = QtCore.QPoint(400, 400)

words = []
# Reduced Categories for Testing
words = [line.rstrip('\n') for line in open('categories.txt')]


class ScribbleArea(QtWidgets.QWidget):

    class InsertLine(QtWidgets.QUndoCommand):
        def __init__(self, index, scribbleArea):
            super().__init__()
            self.index = index
            self.sa = scribbleArea

        def undo(self):
            self.sa.drawImage(self.index)

        def redo(self):
            #self.sa.drawImage(self.index+1)
            pass

    def __init__(self, parent=None):
        super(ScribbleArea, self).__init__(parent)

        self.setAttribute(QtCore.Qt.WA_StaticContents)
        self.setMinimumHeight(450)
        self.setMinimumWidth(530)
        self.modified = False
        self.scribbling = False
        self.myPenWidth = 3
        self.myPenColor = QtCore.Qt.black
        self.image = QtGui.QImage()
        self.lastPoint = QtCore.QPoint()

        self.stack = QtWidgets.QUndoStack()
        self.drawingSegment = []
        self.drawing = []

        # Undo Test
        self.undo = QtWidgets.QPushButton("undo", self)
        self.undo.clicked.connect(self.stack.undo)

        # Deactivate during debug
        #self.gameStart = False

    def setStartPoint(self, startPoint):
        self.lastPoint = startPoint

    def setPenColor(self, newColor):
        self.myPenColor = newColor

    def setPenWidth(self, newWidth):
        self.myPenWidth = newWidth

    def clearImage(self):
        self.image.fill(QtGui.qRgb(255, 255, 255))
        self.modified = True
        self.update()

    def saveImage(self):
        v = qimage2ndarray.recarray_view(self.image)
        return v

    def setMousePos(self, pos):
        if pos == None:
            return
        #QtGui.QCursor.setPos(self.mapToGlobal(QtCore.QPoint(pos[0], pos[1])))

    def buttonEvents(self, report):
        for button in report:
            if button[0] == "B" and button[1]:
                QtTest.QTest.mouseClick(self, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, self.cursor().pos())
                print(self)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            print("test")
            self.lastPoint = event.pos()
            self.scribbling = True

    def mouseMoveEvent(self, event):
        print(event.buttons())
        if (event.buttons() & QtCore.Qt.LeftButton) and self.scribbling:
            self.drawLineTo(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.scribbling:
            self.drawLineTo(event.pos())
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

    def drawImage(self, index):
        self.clearImage()
        for lineSegment in self.drawing[:index]:
            for line in lineSegment:
                if line:
                    self.lastPoint = line.p1()
                    self.drawLineTo(line.p2())
        self.update()
        self.drawing

    def drawLineTo(self, endPoint):
        painter = QtGui.QPainter(self.image)
        painter.setPen(QtGui.QPen(self.myPenColor, self.myPenWidth, QtCore.Qt.SolidLine,
                QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        self.line = QtCore.QLine(self.lastPoint, endPoint)
        self.drawingSegment.append(self.line)
        painter.drawLine(self.line)
        self.modified = True
        rad = self.myPenWidth / 2 + 2
        self.update(QtCore.QRect(self.lastPoint, endPoint).normalized().adjusted(-rad, -rad, +rad, +rad))
        self.lastPoint = QtCore.QPoint(endPoint)

    def resizeImage(self, image, newSize):
        if image.size() == newSize:
            return

        newImage = QtGui.QImage(newSize, QtGui.QImage.Format_RGB32)
        newImage.fill(qRgb(255, 255, 255))
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
        self.drawing.append(self.drawingSegment)
        self.stack.push(self.InsertLine(len(self.drawing)-1, self))
        self.drawingSegment = []


class Painter(QtWidgets.QMainWindow):
    def __init__(self, wiimote, wiiDraw):
        super(Painter, self).__init__()
        self.ui = uic.loadUi("DrawGame.ui", self)
        print(self.pos().x())
        print(self.pos().y())
        self.time = 60
        self.currentWord = ""
        self.gameRunning = True
        self.roundWon = False
        self.roundRunning = False
        self.currentTeam = 2
        self.scoreTeamOne = 0
        self.scoreTeamTwo = 0
        self.guess = ""
        self.initUI()
        self.cw = ScribbleArea(self.ui.frame)
        self.show()
        self.prHelper = helper.QuickDrawHelper()
        self.trainModel = draw.ITTDrawGuesserCNN(self.prHelper.get_num_categories())
        wiimote.buttons.register_callback(self.cw.buttonEvents)
        wiiDraw.register_callback(self.cw.setMousePos)
        wiiDraw.start_processing()

    def initUI(self):
        self.ui.color.clicked.connect(self.setNewColor)
        self.ui.clear.clicked.connect(self.clearImage)
        self.ui.startButton.clicked.connect(self.startGaming)
        self.ui.startGame.clicked.connect(self.startNewRound)
        self.ui.endGame.clicked.connect(self.endGaming)
        self.ui.timer.display(self.time)
        self.ui.team1Score.display(self.scoreTeamOne)
        self.ui.team2Score.display(self.scoreTeamTwo)
        self.ui.blueTeam.hide()

    def startGaming(self):
        self.ui.title.hide()
        self.ui.startButton.hide()
        self.ui.startScreen.lower()
        self.ui.secondText.hide()
        self.time = int(self.ui.selectSeconds.value())
        self.ui.timer.display(self.time)
        self.ui.selectSeconds.hide()
        self.ui.secondsSlider.hide()

    def endGaming(self):
        print("Ende")
        #sys.exit()

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

            self.roundWon = False
            self.roundRunning = True
            self.currentWord = random.choice(words).title()
            self.ui.timer.display(self.time)
            self.ui.category.setText(self.currentWord)
            self.clearImage()

            # Call to draw Line
            #self.cw.setStartPoint(QtCore.QPoint(20,40))
            #self.cw.drawLineTo(QtCore.QPoint(200,200))
            #self.cw.drawLineTo(QtCore.QPoint(300,400))
            #

            t = Thread(target=self.countdown)
            t.start()

    # Just for testing
    def changeGuess(self, guess):
        self.guess = guess
        self.ui.kiGuess.setText("I think it is: %s" % self.guess)

    def countdown(self):
        x = self.time-1
        for i in range(x, -1, -1):
            if i%30 == 0:
                currentImage = self.cw.saveImage()
                self.changeGuess(self.prHelper.get_label(self.trainModel.predict(currentImage)))
            if i%2 == 0:
                self.cw.addSegment()
            if not self.roundWon:
                time.sleep(1)
                self.ui.timer.display(i)
                self.checkGuessing()
            else:
                print("Winner")
                break
        self.processEndRound()
        print("Round End")

    # Todo: Just for Testing / Find better solution!
    def processEndRound(self):
        if self.roundWon:
            if self.currentTeam == 1:
                self.scoreTeamOne += 1
            else:
                self.scoreTeamTwo += 1

        self.ui.team1Score.display(self.scoreTeamOne)
        self.ui.team2Score.display(self.scoreTeamTwo)
        self.checkGameEnd()

    def checkGameEnd(self):
        if self.scoreTeamOne == 3:
            print("Game End, Team 1 won")
            self.gameRunning = False
        elif self.scoreTeamTwo == 3:
            print("Game End, Team 2 won")
            self.gameRunning = False
        self.roundRunning = False

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

    def setKIGuess(self):
        print("Guess:")

def connect_wiimote(btaddr="18:2a:7b:f4:bc:65", attempt=0):
    if len(btaddr) == 17:
        #print("connecting wiimote " + btaddr + "..")
        w = None
        try:
            w = wiimote.connect(btaddr)
        except:
            #print(sys.exc_info())
            pass
        if w is None:
            #print("couldn't connect wiimote. tried it " + str(attempt) + " times")
            time.sleep(3)
            connect_wiimote(btaddr, attempt + 1)
        else:
            #print("succesfully connected wiimote")
            return w
    else:
        #print("bluetooth address has to be 17 characters long")
        return None



def main():
    app = QtWidgets.QApplication(sys.argv)
    wiimote = connect_wiimote()
    wiiDraw = wiimote_drawing.init(wiimote)
    paint = Painter(wiimote, wiiDraw)
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
