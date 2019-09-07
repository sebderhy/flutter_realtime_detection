import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:tflite/tflite.dart';
import 'package:image/image.dart' as ImgLib;

import 'dart:math' as math;

import 'models.dart';

typedef void Callback(List<dynamic> list, int h, int w);

class Camera extends StatefulWidget {
  final List<CameraDescription> cameras;
  final Callback setRecognitions;
  final String model;

  Camera(this.cameras, this.model, this.setRecognitions);

  @override
  _CameraState createState() => new _CameraState();
}

class _CameraState extends State<Camera> {
  CameraController controller;
  bool isDetecting = false;
  bool hasSSimg = false;
  Widget _camPreview;
  Uint8List _imgBytes;
  Image _img;
  Uint8List _lastSSImg;

  // Function taken from https://github.com/flutter/flutter/issues/26348#issuecomment-462321428
  Future<Image> convertYUV420toImageColor(CameraImage image) async {
    try {
      final int width = image.width;
      final int height = image.height;
      final int uvRowStride = image.planes[1].bytesPerRow;
      final int uvPixelStride = image.planes[1].bytesPerPixel;

      print("uvRowStride: " + uvRowStride.toString());
      print("uvPixelStride: " + uvPixelStride.toString());

      // imgLib -> Image package from https://pub.dartlang.org/packages/image
      var img = ImgLib.Image(height, width); // Create Image buffer

      // Fill image buffer with plane[0] from YUV420_888
      for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
          final int uvIndex =
              uvPixelStride * (x / 2).floor() + uvRowStride * (y / 2).floor();
          final int index = y * width + x;
          final int x2 = height - 1 - y;
          final int y2 = width - 1 - x;

          // final int uvIndex = uvPixelStride * (y/2).floor() + uvColStride*(x/2).floor();
          final int index2 = x2 * height + y2;

          final yp = image.planes[0].bytes[index];
          final up = image.planes[1].bytes[uvIndex];
          final vp = image.planes[2].bytes[uvIndex];
          // Calculate pixel color
          int r = (yp + vp * 1436 / 1024 - 179).round().clamp(0, 255);
          int g = (yp - up * 46549 / 131072 + 44 - vp * 93604 / 131072 + 91)
              .round()
              .clamp(0, 255);
          int b = (yp + up * 1814 / 1024 - 227).round().clamp(0, 255);
          // color: 0x FF  FF  FF  FF
          //           A   B   G   R
          img.data[index2] = (0xFF << 24) | (b << 16) | (g << 8) | r;
        }
      }

      ImgLib.PngEncoder pngEncoder = new ImgLib.PngEncoder(level: 0, filter: 0);
      List<int> png = pngEncoder.encodeImage(img);
      // muteYUVProcessing = false;
      return Image.memory(png,
          height: width.toDouble(), width: height.toDouble());
    } catch (e) {
      print(">>>>>>>>>>>> ERROR:" + e.toString());
    }
    return null;
  }

  @override
  void initState() {
    super.initState();

    if (widget.cameras == null || widget.cameras.length < 1) {
      print('No camera is found');
    } else {
      controller = new CameraController(
        widget.cameras[0],
        ResolutionPreset.medium,
      );
      controller.initialize().then((_) {
        if (!mounted) {
          return;
        }
        setState(() {});

        controller.startImageStream((CameraImage img) {
          if (!isDetecting) {
            isDetecting = true;

            int startTime = new DateTime.now().millisecondsSinceEpoch;

            if (widget.model == mobilenet) {
              Tflite.runModelOnFrame(
                bytesList: img.planes.map((plane) {
                  return plane.bytes;
                }).toList(),
                imageHeight: img.height,
                imageWidth: img.width,
                numResults: 2,
              ).then((recognitions) {
                int endTime = new DateTime.now().millisecondsSinceEpoch;
                print("Detection took ${endTime - startTime}");

                widget.setRecognitions(recognitions, img.height, img.width);

                isDetecting = false;
              });
            } else if (widget.model == flowers) {
              print("flower model chosen !!");
              Tflite.runModelOnFrame(
                bytesList: img.planes.map((plane) {
                  return plane.bytes;
                }).toList(),
                imageHeight: img.height,
                imageWidth: img.width,
                // imageMean: 127.5,
                // imageStd: 127.5,
                numResults: 2,
              ).then((recognitions) {
                print("Flower classification obtained");
                int endTime = new DateTime.now().millisecondsSinceEpoch;
                print("Detection took ${endTime - startTime}");

                widget.setRecognitions(recognitions, img.height, img.width);
                print("setRecognitions passed");

                isDetecting = false;
                // setState(() {});
              });
            } else if (widget.model == posenet) {
              Tflite.runPoseNetOnFrame(
                bytesList: img.planes.map((plane) {
                  return plane.bytes;
                }).toList(),
                imageHeight: img.height,
                imageWidth: img.width,
                numResults: 2,
              ).then((recognitions) {
                int endTime = new DateTime.now().millisecondsSinceEpoch;
                print("Detection took ${endTime - startTime}");

                widget.setRecognitions(recognitions, img.height, img.width);

                isDetecting = false;
              });
            } else if (widget.model == deeplab) {
              Tflite.runSegmentationOnFrame(
                      bytesList: img.planes.map((plane) {
                        return plane.bytes;
                      }).toList(),
                      imageHeight: img.height,
                      imageWidth: img.width,
                      imageMean: 127.5, // defaults to 0.0
                      imageStd: 127.5, // defaults to 255.0
                      outputType: "png",
                      rotation: 180)
                  .then((ssImage) {
                int endTime = new DateTime.now().millisecondsSinceEpoch;
                print("SemSeg took ${endTime - startTime}");
                setState(() {
                  convertYUV420toImageColor(img).then((onImg) {
                    _img = onImg;
                  });
                  _lastSSImg = ssImage;
                  hasSSimg = true;
                });
                print("Semantic Segmentation Done !");

                isDetecting = false;
              });
            } else {
              Tflite.detectObjectOnFrame(
                bytesList: img.planes.map((plane) {
                  return plane.bytes;
                }).toList(),
                model: widget.model == yolo ? "YOLO" : "SSDMobileNet",
                imageHeight: img.height,
                imageWidth: img.width,
                imageMean: widget.model == yolo ? 0 : 127.5,
                imageStd: widget.model == yolo ? 255.0 : 127.5,
                numResultsPerClass: 1,
                threshold: widget.model == yolo ? 0.2 : 0.4,
              ).then((recognitions) {
                int endTime = new DateTime.now().millisecondsSinceEpoch;
                print("Detection took ${endTime - startTime}");

                widget.setRecognitions(recognitions, img.height, img.width);

                isDetecting = false;
              });
            }
          }
        });
      });
    }
  }

  @override
  void dispose() {
    controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (controller == null || !controller.value.isInitialized) {
      return Container();
    }

    // _camPreview = CameraPreview(controller);
    var tmp = MediaQuery.of(context).size;
    var screenH = math.max(tmp.height, tmp.width);
    var screenW = math.min(tmp.height, tmp.width);
    tmp = controller.value.previewSize;
    var previewH = math.max(tmp.height, tmp.width);
    var previewW = math.min(tmp.height, tmp.width);
    var screenRatio = screenH / screenW;
    var previewRatio = previewH / previewW;
    // var child1 = CameraPreview(controller);
    var child2;

    if (hasSSimg) {
      print("*** We have an SS image !!");
      final ssImageToDisplay = DecorationImage(
          alignment: Alignment.topCenter,
          image: MemoryImage(_lastSSImg),
          fit: BoxFit.fill);

      // final rotatedWidget = Transform.rotate(
      //       child: ssImageToDisplay,
      //     )
      // final rotatedWidget = RotatedBox(
      //   quarterTurns: 2,
      //   child: ssImageToDisplay,
      // );

      // child2 = Container(
      //   decoration: BoxDecoration(image: ssImageToDisplay),
      //   child: Opacity(opacity: 0.3, child: _img),
      // );

      print("_img.width = " + _img.width.toString());
      print("_img.height = " + _img.height.toString());
      final ssImg = Image.memory(_lastSSImg,
          width: _img.height.toDouble(),
          height: _img.width.toDouble(),
          fit: BoxFit.fill);

      print("ssImg.width = " + ssImg.width.toString());
      print("ssImg.height = " + ssImg.height.toString());
      child2 = Stack(
        alignment: Alignment.topLeft,
        fit: StackFit.passthrough,
        children: [ssImg, Opacity(opacity: 0.3, child: _img)],
      );

      // child2 = Image.memory(_lastSSImg);

    }

    return OverflowBox(
        maxHeight: screenRatio > previewRatio
            ? screenH
            : screenW / previewW * previewH,
        maxWidth: screenRatio > previewRatio
            ? screenH / previewH * previewW
            : screenW,
        child: hasSSimg ? child2 : CameraPreview(controller));
  }
}
