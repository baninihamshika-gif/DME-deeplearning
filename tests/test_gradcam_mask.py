"""Grad-CAM display masking (app.py): the heatmap must not light up pixels
that are not part of the scan -- letterbox padding, flattening zero fill, or
blank fill baked into the source file."""

import numpy as np

import config
from utils import flatten_retina, preprocess_image, source_fill_mask


def _synthetic_scan(h=200, w=400):
    rng = np.random.default_rng(0)
    img = rng.integers(20, 60, size=(h, w)).astype(np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    band = np.abs(yy - (100 + 0.0008 * (xx - w / 2) ** 2)) < 15  # curved bright retina
    img[band] = 200
    img[h - 12 :, : w // 3] = 255  # white source-fill wedge on the bottom edge
    return img


def test_default_return_unchanged():
    img = _synthetic_scan()
    out = preprocess_image(img, image_size=config.IMAGE_SIZE)
    assert len(out) == 3
    rgb, bbox, fallback, valid = preprocess_image(img, image_size=config.IMAGE_SIZE, return_valid_mask=True)
    assert np.array_equal(out[0], rgb) and out[1] == bbox and out[2] == fallback
    assert valid.shape == rgb.shape[:2] and valid.dtype == bool


def test_valid_mask_excludes_padding_and_fill():
    img = _synthetic_scan()
    rgb, (y0, y1, x0, x1), _, valid = preprocess_image(img, image_size=config.IMAGE_SIZE, return_valid_mask=True)
    outside = np.ones_like(valid)
    outside[y0:y1, x0:x1] = False
    assert not valid[outside].any()  # letterbox padding
    assert valid[y0:y1, x0:x1].mean() > 0.8  # most of the scan stays
    assert source_fill_mask(img)[-1, 0] and not source_fill_mask(img)[100, 200]


def test_flatten_valid_marks_zero_fill():
    img = _synthetic_scan()
    flat, fallback, valid = flatten_retina(img, return_valid=True)
    flat2, fallback2 = flatten_retina(img)
    assert np.array_equal(flat, flat2) and fallback == fallback2
    if not fallback:
        assert (valid == 0).any()
        assert (flat[valid == 0] == 0).all()


def test_mask_cam_to_image():
    from app import mask_cam_to_image

    cam = np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10)
    valid = np.zeros((10, 10), bool)
    valid[:5] = True
    out = mask_cam_to_image(cam, valid)
    assert (out[~valid] == 0).all()
    assert np.isclose(out[valid].max(), 1.0)
    assert np.allclose(out[valid], cam[valid] / cam[valid].max())


def test_generate_app_cam_shape_range_and_target():
    import timm
    import torch

    import train
    from app import GRADCAM_BLOCK, _LogitDifference, generate_app_cam

    model = timm.create_model(train.MODEL_NAME, pretrained=False, num_classes=len(config.CLASS_NAMES)).eval()
    assert len(model.blocks) > GRADCAM_BLOCK
    x = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    for cls in (0, 1):
        cam = generate_app_cam(model, x, cls)
        assert cam.shape == (config.IMAGE_SIZE, config.IMAGE_SIZE)
        assert cam.min() >= -1e-6 and cam.max() <= 1 + 1e-6
    logits = torch.tensor([[0.2, 1.5]])
    assert torch.isclose(_LogitDifference(1)(logits), torch.tensor([1.3])).all()
    assert torch.isclose(_LogitDifference(0)(logits), torch.tensor([-1.3])).all()
