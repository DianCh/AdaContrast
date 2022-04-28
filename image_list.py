import os
from PIL import Image
from torch.utils.data import Dataset


def load_image(img_path):
    img = Image.open(img_path)
    img = img.convert("RGB")
    return img


class ImageList(Dataset):
    def __init__(
        self,
        image_root: str,
        label_file: str,
        transform=None,
        pseudo_item_list=None,
    ):
        self.image_root = image_root
        self._label_file = label_file
        self.transform = transform

        assert (
            label_file or pseudo_item_list
        ), f"Must provide either label file or pseudo labels."
        self.item_list = (
            self.build_index(label_file) if label_file else pseudo_item_list
        )

    def build_index(self, label_file):
        """Build a list of <image path, class label> items.

        Args:
            label_file: path to the domain-net label file

        Returns:
            item_list: a list of <image path, class label> items.
        """
        # read in items; each item takes one line
        with open(label_file, "r") as fd:
            lines = fd.readlines()
        lines = [line.strip() for line in lines if line]

        item_list = []
        for item in lines:
            img_file, label = item.split()
            img_path = os.path.join(self.image_root, img_file)
            label = int(label)
            item_list.append((img_path, label, img_file))

        return item_list

    def __getitem__(self, idx):
        """Retrieve data for one item.

        Args:
            idx: index of the dataset item.
        Returns:
            img: <C, H, W> tensor of an image
            label: int or <C, > tensor, the corresponding class label. when using raw label
                file return int, when using pseudo label list return <C, > tensor.
        """
        img_path, label, _ = self.item_list[idx]
        img = load_image(img_path)
        if self.transform:
            img = self.transform(img)

        return img, label, idx

    def __len__(self):
        return len(self.item_list)
