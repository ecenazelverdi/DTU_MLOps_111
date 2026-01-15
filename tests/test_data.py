import os

def test_drone_segmentation_dataset_files():
    # Test that there are 598 files in each of the nnUNet_raw/Dataset101_DroneSeg folders
    base_path = 'nnUNet_raw/Dataset101_DroneSeg'
    images_path = os.path.join(base_path, 'imagesTr')
    labels_path = os.path.join(base_path, 'labelsTr')
    images_files = os.listdir(images_path)
    labels_files = os.listdir(labels_path)
    assert len(images_files) == 1200, f"Expected 1200 image files, found {len(images_files)}"
    assert len(labels_files) == 400, f"Expected 1200 label files, found {len(labels_files)}"

def test_json_files():
    # Test that the dataset JSON files exist
    base_path = 'nnUNet_raw/Dataset101_DroneSeg'
    dataset_json_path = os.path.join(base_path, 'dataset.json')
    assert os.path.isfile(dataset_json_path), f"Dataset JSON file not found at {dataset_json_path}"