# Archived: v1 capacity/LR roster (resnet + resnext families)

19 models trained under the original 100-config roster (10 families x 10
variants, varying encoder capacity + LR, fixed 512px resolution). Superseded
by the 120-config depth x resolution grid in `roster_cls.py` (10 families x 4
depth tiers x 3 resolution tiers), which drops resnext/seresnet/mobilenet and
adds xception/maxvit/coatnet (xception replaces the originally-planned
"Inception" slot - classic Inception-v3/Inception-ResNet-v2/InceptionNeXt were
all considered and rejected, see roster_cls.py's architecture note for why).
`_meta/` holds the old `skip_models_cls.json`
and the one failure log (`cls_resnext_06.txt`, a transient HuggingFace Hub
download error on `skresnext50_32x4d` - still on the debug backlog, unrelated
to this archival).

Kept for reference, not deleted - nothing here feeds into the new roster.
