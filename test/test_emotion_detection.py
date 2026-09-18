import unittest
from EmotionDetection.emotion_detection import emotion_detector


class TestEmotionDetector(unittest.TestCase):
    """Test the emotion detector function."""

    def test_joy(self):
        """Test joy detection."""
        self.assertEqual(
            emotion_detector("I am glad this happened")["dominant_emotion"],
            "joy"
        )

    def test_anger(self):
        """Test anger detection."""
        self.assertEqual(
            emotion_detector("I am really mad about this")["dominant_emotion"],
            "anger"
        )

    def test_disgust(self):
        """Test disgust detection."""
        self.assertEqual(
            emotion_detector(
                "I feel disgusted just hearing about this"
            )["dominant_emotion"],
            "disgust"
        )

    def test_sadness(self):
        """Test sadness detection."""
        self.assertEqual(
            emotion_detector("I am so sad about this")["dominant_emotion"],
            "sadness"
        )

    def test_fear(self):
        """Test fear detection."""
        self.assertEqual(
            emotion_detector(
                "I am really afraid that this will happen"
            )["dominant_emotion"],
            "fear"
        )


if __name__ == "__main__":
    unittest.main()
