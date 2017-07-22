import sys, random, sched, time
from threading import Thread
import numpy as np

from PyQt5 import uic, QtWidgets, QtCore, QtGui, QtPrintSupport
from PyQt5.QtGui import qRgb

p1 = QtCore.QPoint(0, 0)
p2 = QtCore.QPoint(400, 400)

words = []
# Reduced Categories for Testing
with open('categories.txt') as f:
    words = f.read().split()

class InsertLine(QtWidgets.QUndoCommand):
    def __init__(self, line, arrayList):
            super().__init__()
            self.line = line
            self.arrayList = arrayList
    def undo(self):
        self.arrayList.pop()

    def redo(self):
        self.arrayList.append(self.line)



class ScribbleArea(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(ScribbleArea, self).__init__(parent)

        self.setAttribute(QtCore.Qt.WA_StaticContents)
        self.setMinimumHeight(450)
        self.setMinimumWidth(530)
        self.modified = False
        self.scribbling = False
        self.myPenWidth = 1
        self.myPenColor = QtCore.Qt.blue
        self.image = QtGui.QImage()
        self.lastPoint = QtCore.QPoint()
        self.stack = QtWidgets.QUndoStack()
        self.lineList = []

        # Undo Test
        #self.undo = QtWidgets.QPushButton("undo", self)
        #self.undo.clicked.connect(self.stack.undo)
        #self.redo = QtWidgets.QPushButton("redo", self)
        #self.redo.clicked.connect(self.stack.redo)
        #self.redo.move(0, 50)

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

    # Vll undo? Saves the current painter state (pushes the state onto a stack). A save() must be followed by a corresponding restore(); the end() function unwinds the stack.
    def saveImage(self, fileName, fileFormat):
        visibleImage = self.image
        self.resizeImage(visibleImage, self.size())

        #ptr = visibleImage.bits()
        #ptr.setsize(visibleImage.byteCount())
        #arr = np.asarray(ptr).reshape(visibleImage.height(), visibleImage.width(), 4)
        #print(arr)
        #print(QtGui.QImageWriter.supportedImageFormats())



        if visibleImage.save(fileName, fileFormat):
            self.modified = False
            return True
        else:
            return False

    # Just to Test Drawing####
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.lastPoint = event.pos()
            self.scribbling = True

    def mouseMoveEvent(self, event):
        if (event.buttons() & QtCore.Qt.LeftButton) and self.scribbling:
            self.drawLineTo(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.scribbling:
            self.drawLineTo(event.pos())
            self.scribbling = False

    ###############################

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

    def drawLineTo(self, endPoint):
        painter = QtGui.QPainter(self.image)
        painter.setPen(QtGui.QPen(self.myPenColor, self.myPenWidth, QtCore.Qt.SolidLine,
                QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
        self.line = QtCore.QLine(self.lastPoint, endPoint)
        command = InsertLine(self.line, self.lineList)
        self.stack.push(command)

        # Undo Test with Array
        #print(len(self.lineList))

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


class Painter(QtWidgets.QMainWindow):
    def __init__(self):
        super(Painter, self).__init__()
        self.ui = uic.loadUi("DrawGame.ui", self)
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
            self.saveFile('png')
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
            if not self.roundWon:
                time.sleep(1)
                self.changeGuess(random.choice(words).title())
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


def main():
    app = QtWidgets.QApplication(sys.argv)
    paint = Painter()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
