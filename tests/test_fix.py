
import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Adjust path so we can import modules from src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Mock the imports that might fail or are not needed for this specific test
sys.modules['pydub'] = MagicMock()

class TestProcessFolderBase(unittest.TestCase):
    @patch('os.listdir')
    @patch('os.path.exists')
    @patch('os.makedirs')
    def test_process_folder_base_calls_correctly(self, mock_makedirs, mock_exists, mock_listdir):

        # Setup mocks
        mock_listdir.return_value = ['test.mp3']
        mock_exists.return_value = True

        # Mocking audio_bchampion.forced_alignment
        mock_fa = MagicMock()
        # transcribe_audio is removed from usage, so we don't strictly need to mock its return value for logic,
        # but the import might still be there.
        mock_fa.get_dialogue_segments.return_value = [(1.0, 2.0)]
        sys.modules['audio_bchampion.forced_alignment'] = mock_fa

        # Mocking audio_bchampion.audio_processing
        mock_ap = MagicMock()
        sys.modules['audio_bchampion.audio_processing'] = mock_ap

        # Import main after setting up sys.modules
        import main

        # Run the function
        main.process_folder_base('input_dir', 'ebook.txt', 'output_dir')

        # Verify process_audiobook was called with 3 arguments
        # args: audio_path, dialogue_intervals, out_dir

        mock_ap.process_audiobook.assert_called_once()
        args, kwargs = mock_ap.process_audiobook.call_args

        print(f"Called with args: {args}")

        self.assertEqual(len(args), 3, "process_audiobook should be called with 3 arguments")
        self.assertEqual(args[0], os.path.join('input_dir', 'test.mp3'))
        self.assertEqual(args[1], [(1.0, 2.0)])
        self.assertEqual(args[2], os.path.join('output_dir', 'test'))

if __name__ == '__main__':
    unittest.main()
