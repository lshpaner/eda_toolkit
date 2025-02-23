import pytest
import os
from unittest import mock

from eda_toolkit import print_art


def test_print_art_specific(capfd):
    """Test printing a specific ASCII art piece."""
    print_art("eda_toolkit_logo")
    captured = capfd.readouterr()
    assert "eda_toolkit_logo" in captured.out  # Ensure label is printed
    assert (
        "+--------------------------------------------------------------------------------+"
        in captured.out
    )  # Part of the ASCII art


def test_print_art_all(capfd):
    """Test printing all ASCII art."""
    print_art(all=True)
    captured = capfd.readouterr()
    assert "eda_toolkit_logo" in captured.out  # Ensure multiple arts are printed
    assert "royce_hall_bb" in captured.out  # Another random one to verify


def test_print_art_suffix_filter(capfd):
    """Test filtering ASCII art by suffix."""
    print_art(suffix="bb")
    captured = capfd.readouterr()
    assert "royce_hall_bb" in captured.out  # Should be printed
    assert "royce_hall_wb" not in captured.out  # Should not be printed


def test_print_art_invalid_name(capfd):
    """Test handling an invalid ASCII art name."""
    print_art("invalid_art_name")
    captured = capfd.readouterr()

    # Instead of an exact match, check if the key phrase is within the output
    assert "'invalid_art_name' not found" in captured.out
    assert (
        "Available options are:" in captured.out
    )  # Ensures list of options is printed


def test_print_art_no_input(capfd):
    """Test no input scenario (lists available options)."""
    print_art()
    captured = capfd.readouterr()
    assert "Available options are:" in captured.out


def test_print_art_save_output(tmp_path):
    """Test saving output to a file."""
    output_file = "test_output.txt"
    output_path = tmp_path / "output_dir"

    print_art("eda_toolkit_logo", output_file=output_file, output_path=str(output_path))

    output_file_path = output_path / output_file
    assert output_file_path.exists()

    with open(output_file_path, "r") as file:
        content = file.read()

    assert "eda_toolkit_logo" in content
    assert (
        "+--------------------------------------------------------------------------------+"
        in content
    )  # Ensure ASCII content is written


def test_print_art_conflicting_args():
    """Test handling of conflicting arguments (all=True with specific art_names)."""
    with pytest.raises(
        ValueError, match="You cannot specify both `all=True` and specific `art_names`"
    ):
        print_art("eda_toolkit_logo", all=True)


def test_print_art_directory_creation(mocker, tmp_path):
    """Test directory creation when saving output."""
    output_file = "test_output.txt"
    output_path = tmp_path / "new_directory"

    mock_makedirs = mocker.patch("os.makedirs", wraps=os.makedirs)

    print_art("eda_toolkit_logo", output_file=output_file, output_path=str(output_path))

    mock_makedirs.assert_called_once_with(
        str(output_path), exist_ok=True
    )  # Ensure directory was created
    assert (output_path / output_file).exists()  # File should exist


def test_print_art_auto_txt_extension(tmp_path):
    """Ensure .txt is added automatically if no extension is provided."""
    output_file = "test_output"  # No extension
    output_path = tmp_path / "output_dir"

    print_art("eda_toolkit_logo", output_file=output_file, output_path=str(output_path))

    output_file_path = output_path / (
        output_file + ".txt"
    )  # Should have .txt extension
    assert output_file_path.exists()


def test_print_art_suffix_no_matches(capfd):
    """Ensure suffix filtering returns an empty list when no matches are found."""
    print_art(suffix="nonexistent_suffix")
    captured = capfd.readouterr()
    assert (
        "No keys found with suffix 'nonexistent_suffix'." in captured.out
    )  # Fix wording


def test_print_art_invalid_output_path(mocker):
    """Ensure ValueError is raised when output path is invalid."""
    mocker.patch("os.makedirs", side_effect=OSError("Permission denied"))

    with pytest.raises(OSError, match="Permission denied"):
        print_art(
            "eda_toolkit_logo", output_file="output.txt", output_path="/invalid/path"
        )


def test_print_art_save_file_contents(tmp_path):
    """Ensure saved file contains the correct ASCII art."""
    output_file = "test_output.txt"
    output_path = tmp_path / "output_dir"

    print_art("eda_toolkit_logo", output_file=output_file, output_path=str(output_path))

    output_file_path = output_path / output_file
    assert output_file_path.exists()

    with open(output_file_path, "r") as file:
        content = file.read()

    assert "eda_toolkit_logo" in content
    assert (
        "+--------------------------------------------------------------------------------+"
        in content
    )  # ASCII border


def test_print_art_suffix_and_all(capfd):
    """Ensure that specifying `suffix` and `all=True` does not override `suffix`."""
    print_art(all=True, suffix="bb")
    captured = capfd.readouterr()
    assert (
        "No keys found with suffix 'bb'." not in captured.out
    )  # Ensure some output exists
