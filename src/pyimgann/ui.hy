(import [numpy :as np]
        [cv2]
        [PyQt4.QtGui :as qt])

(defn square [x] (* x x))

(defclass OpenCVQImage [qt.QImage]
  []
  (defn --init-- [self img]
    (let [depth (.depth img)
          nChannels (.nChannels img)]
      (print "opencvqimage:" depth nChannels))))
