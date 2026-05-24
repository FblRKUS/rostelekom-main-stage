import pytest
from unittest.mock import patch, MagicMock
from index import main

@patch("index.Path")
@patch("index.argparse.ArgumentParser.parse_args")
@patch("index.CodeIndexer")
@patch("index.VectorStore")
def test_index_directory(mock_vs_class, mock_indexer_class, mock_parse_args, mock_path_class):
    mock_parse_args.return_value = MagicMock(
        path="some_dir",
        github=None
    )
    
    mock_path_instance = MagicMock()
    mock_path_instance.is_dir.return_value = True
    mock_path_class.return_value.resolve.return_value = mock_path_instance
    
    mock_indexer = MagicMock()
    mock_indexer_class.return_value = mock_indexer
    mock_indexer.scan_directory.return_value = [{"chunk_id": "c1", "content": "func()"}]
    
    mock_vs = MagicMock()
    mock_vs_class.return_value = mock_vs
    
    main()
    
    mock_indexer_class.assert_called_once()
    mock_indexer.scan_directory.assert_called_with(mock_path_instance)
    mock_vs_class.assert_called_once()
    mock_vs.add_chunks.assert_called_once()
    
@patch("index.argparse.ArgumentParser.parse_args")
@patch("index._download_github_archive")
@patch("index.CodeIndexer")
@patch("index.VectorStore")
def test_index_github(mock_vs_class, mock_indexer_class, mock_download, mock_parse_args):
    mock_parse_args.return_value = MagicMock(
        path=None,
        github="https://github.com/owner/repo"
    )
    
    mock_download.return_value = "temp_dir/repo-main"
    
    mock_indexer = MagicMock()
    mock_indexer_class.return_value = mock_indexer
    mock_indexer.scan_directory.return_value = [{"chunk_id": "c1", "content": "func()"}]
    
    mock_vs = MagicMock()
    mock_vs_class.return_value = mock_vs
    
    main()
    
    mock_download.assert_called_once()
    mock_indexer.scan_directory.assert_called_with("temp_dir/repo-main")
    mock_vs.add_chunks.assert_called_once()

def test_index_no_args(capsys):
    with patch("sys.argv", ["index.py"]):
        with pytest.raises(SystemExit):
            main()
