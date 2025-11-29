
import unittest
import sys
import os
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath('src'))
sys.path.insert(0, os.getcwd())

class TestMainLogic(unittest.TestCase):

    def test_process_folder_alternate_success(self):
        # Manually inject the mock module into sys.modules BEFORE starting the test
        # This prevents ModuleNotFoundError when main.py (or we) try to import it.
        mock_forced_alignment = MagicMock()
        mock_audio_bchampion = MagicMock()
        mock_audio_bchampion.forced_alignment = mock_forced_alignment

        # We need to ensure 'audio_bchampion' is in sys.modules
        sys.modules['audio_bchampion'] = mock_audio_bchampion
        sys.modules['audio_bchampion.forced_alignment'] = mock_forced_alignment

        # Setup return values for our manual mock
        mock_forced_alignment.process_transcription_alternate.return_value = (
            [], # dialogue_subtitles (list)
            [], # narration_subtitles (list)
            100.0, # total_duration (float)
            {} # speaker_assignments (dict)
        )

        # Mock os.listdir and os.makedirs using patch context managers
        with patch('os.listdir') as mock_listdir, \
             patch('os.makedirs') as mock_makedirs:

            mock_listdir.return_value = ['test.mp3']

            # Import the function to test
            from src.main import process_folder_alternate

            # Attempt to run process_folder_alternate
            try:
                process_folder_alternate('input', 'ebook.txt', 'output')
            except Exception as e:
                self.fail(f"process_folder_alternate raised exception: {e}")

            # Verify process_audiobook_alternate was called with correct arguments
            mock_forced_alignment.process_audiobook_alternate.assert_called_once()
            args, kwargs = mock_forced_alignment.process_audiobook_alternate.call_args

            # Args: audio_path, dialogue_subtitles, narration_subtitles, speaker_assignments, out_dir
            # Note: inputs are 'input' and 'test.mp3' so joined path is 'input/test.mp3' (or backslash on windows, but os.path.join handles it)
            self.assertEqual(args[0], os.path.join('input', 'test.mp3'))
            self.assertEqual(args[1], [])
            self.assertEqual(args[2], [])
            self.assertEqual(args[3], {})
            self.assertEqual(args[4], os.path.join('output', 'test'))

if __name__ == '__main__':
    unittest.main()
