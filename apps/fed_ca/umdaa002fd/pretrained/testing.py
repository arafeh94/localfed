from apps.fed_ca.umdaa002fd.pretrained.inception_resnet_v1 import InceptionResnetV1

vggface2 = InceptionResnetV1(pretrained='vggface2').eval()

print(vggface2)

