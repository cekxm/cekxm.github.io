---
layout: post
title: "Android Opencv Ndk"
date: 2024-12-30 00:18:07 +0800
categories: []
description: 简要介绍
tags: 
thumbnail: 
toc:
  sidebar: left
typora-root-url: ../
---


## Android studio 安装

1. 一般是下载最新稳定版，比如目前的最新稳定版是 
2. **不用提前安装 JDK 和 JRE**，android studio 自带，**也不用再设置环境变量 JAVA_HOME, CLASSPATH, PATH**
3. 安装，选默认设置，会安装很多东西（settings -> system settings -> Android SDK），需要提前准备大量剩余空间
   - SDK Platforms（就是 API level，和安卓版本号不同），可以安装比较新的，比如33。而 minSdkVersion 决定了可以安装的最低 API，NDK 要求最低为 21
      - System image 这个是为了建立 AVD模拟器需要的，[有各种版本区别](https://developer.android.google.cn/studio/releases/platforms?hl=zh-cn)
   - SDK Tools，版本号和 sdk platforms 版本号不一定要对应，可以安装最新版本，可能随项目要求它会自动下载一些其他版本，随缘吧

### 安装 NDK

可以安装最新版本，可能随项目要求它会自动下载一些其他版本，随缘吧

<img class="img-fluid" src="/images/2024-12-30-android-opencv-ndk/image-20230721185423109.png" alt="image-20230721185423109" style="zoom: 50%;" />

<img class="img-fluid" src="/images/2024-12-30-android-opencv-ndk/image-20230721185451551.png" alt="image-20230721185451551" style="zoom:50%;" />

<img class="img-fluid" src="/images/2024-12-30-android-opencv-ndk/image-20230721185508339.png" alt="image-20230721185508339" style="zoom:50%;" />

<img class="img-fluid" src="/images/2024-12-30-android-opencv-ndk/image-20230721185533927.png" alt="image-20230721185533927" style="zoom:50%;" />

### 其他：

#### [解决Android Studio Build Output 输出的中文显示乱码](https://blog.csdn.net/weixin_43782998/article/details/119718959)

## 安装及配置 OpenCV android sdk

### 参考资料

这一部分是比较麻烦的，主要参考以下网页：

1. [Android OpenCV4.8.0のカメラプレビュー（tutorialを基に）](https://coskxlabsite.stars.ne.jp/html/android/OpenCVpreview/OpenCVpreview_A.html)
2. [OpenCV On Android最佳环境配置指南(Android Studio篇)](https://juejin.cn/post/6972167812445896735)
3. [OpenCV 在 Android Studio 的使用教程](https://blog.csdn.net/qq_41885673/article/details/115324283)

配置过程

cv::FaceDetectorYN cv::FaceRecognizerSF 要求最低 4.5.4，本次选用 opencv-4.5.5-android-sdk.zip

选择更高版本应该没有问题。

配置的过程，就是建立project的过程。

### new project->native C++

![image-20230721190740698](/images/2024-12-30-android-opencv-ndk/image-20230721190740698.png){: .img-fluid}

### file->new->import module

![image-20230721191139114](/images/2024-12-30-android-opencv-ndk/image-20230721191139114.png){: .img-fluid}

注意是已 sdk 为结尾

#### Kotlin 问题

现在还没导入成功，会报错 > Plugin with id 'kotlin-android' not found.

因为这一步会把整个 sdk 文件夹拷贝到工程目录，所以注意修改工程目录/opencv下的 gradle，而不是被拷贝的那个库的文件。因为还没导入成功，所以在 android 页还看不到 opencv，切换到 project 页

![image-20230721191625330](/images/2024-12-30-android-opencv-ndk/image-20230721191625330.png){: .img-fluid}

注释 apply plugin: 'kotlin-android'

然后在 build.gradle(Project: xxxx) 添加

```
buildscript {
    dependencies {
        classpath "org.jetbrains.kotlin:kotlin-gradle-plugin:1.7.20"
    }
}
```

注意这边的版本可以用 everything 找一下，和电脑中下载的要相同。

#### namespace 问题

做到这一步，如果编译，会提示 Namespace not specified. 

修改 opencv/build.gradle，增加一句话

```
android {
    namespace 'org.opencv'
    compileSdkVersion 26
```

opencv 这儿 sdk 版本是26。在此阶段，最好将 Gradle Script/build.gradle （Module OpenCV） 中的 compileSdkVersion targetSdkVersion 更改为和app一样的版本

![image-20230721193557426](/images/2024-12-30-android-opencv-ndk/image-20230721193557426.png){: .img-fluid}

现在 Android 页中可以看到 opencv 出来了，则导入算成功了。

### 为app添加opencv module依赖

选中app，Open module settings -> Dependencies

为app添加 module依赖，选择 Opencv

## App 的 Java 部分

接下来我们要建一个通用的 activity 和 layout。后面这部分就尽量不修改。

### AndroidManifest.xml文件中添加权限

主要有摄像头权限、存储权限

```xml
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:tools="http://schemas.android.com/tools">

    <uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />
    <uses-permission android:name="android.permission.CAMERA" />

    <uses-feature
        android:name="android.hardware.camera"
        android:required="false" />
    <uses-feature
        android:name="android.hardware.camera.autofocus"
        android:required="false" />
    <uses-feature
        android:name="android.hardware.camera.front"
        android:required="false" />
    <uses-feature
        android:name="android.hardware.camera.front.autofocus"
        android:required="false" />
```

#### 问题1：org.opencv.engine 不存在

此时编译会碰到 OpenCVEngineInterface 错误了：程序包org.opencv.engine不存在

它是 opencv android sdk 的问题。

打开 opencv build.gradle，在 externalNativeBuild后面添加 buildFeatures

```xml
    externalNativeBuild {
        cmake {
            path (project.projectDir.toString() + '/libcxx_helper/CMakeLists.txt')
        }
    }

    buildFeatures {
        aidl true
    }
```

#### 问题2：找不到 org.opencv.BuildConfig

在 gradle.properties 中增加一句话

```
android.defaults.buildfeatures.buildconfig=true
```

![image-20230721200131762](/images/2024-12-30-android-opencv-ndk/image-20230721200131762.png){: .img-fluid}

### activity_main.xml Layout 

```xml
<?xml version="1.0" encoding="utf-8"?>
<androidx.constraintlayout.widget.ConstraintLayout xmlns:android="http://schemas.android.com/apk/res/android"
    xmlns:app="http://schemas.android.com/apk/res-auto"
    xmlns:tools="http://schemas.android.com/tools"
    android:layout_width="match_parent"
    android:layout_height="match_parent"
    tools:context=".MainActivity">

    <org.opencv.android.JavaCameraView
        android:id="@+id/javaCameraView"
        android:layout_width="match_parent"
        android:layout_height="match_parent"
        app:camera_id="back"
        app:show_fps="true" />

</androidx.constraintlayout.widget.ConstraintLayout>
```

### MainActivity.java

```java
package com.example.cvapp;

import androidx.appcompat.app.AppCompatActivity;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Collections;
import java.util.List;
import java.util.ArrayList;

import android.os.Bundle;
import android.widget.TextView;
import android.view.SurfaceView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.imgproc.Imgproc;
import org.opencv.android.JavaCameraView;

import com.example.cvapp.databinding.ActivityMainBinding;

public class MainActivity extends CameraActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    // Used to load the 'cvapp' library on application startup.
    static {
        System.loadLibrary("cvapp");
    }
    private JavaCameraView javaCameraView;

    private BaseLoaderCallback baseLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    javaCameraView.enableView();
                }
                break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    @Override
    protected List<? extends CameraBridgeViewBase> getCameraViewList() {
        List<CameraBridgeViewBase> list = new ArrayList<>();
        list.add(javaCameraView);
        return list;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        javaCameraView = findViewById(R.id.javaCameraView);
        javaCameraView.setVisibility(SurfaceView.VISIBLE);
        javaCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause() {
        super.onPause();
        if (javaCameraView != null) {
            javaCameraView.disableView();
        }
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, baseLoaderCallback);
        } else {
            baseLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {

    }

    @Override
    public void onCameraViewStopped() {

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        return inputFrame.gray();
    }
}
```

注意：

1. **此时可以可以运行程序了**，这是一个播放摄像头视频的程序，并把图像变成黑白
2. 但是现在是纯java的，还没涉及到 c++ 部分
3. 代码里面有几行，是引入 c++ 库，但是实际上这个时候还没用到

```java
    static {
        System.loadLibrary("cvapp");
    }
```

## OpenCV NDK 部分

### 配置

OpenCV-android-sdk\sdk\build.gradle 这个文件前头的注释很重要，推荐详细阅读。

```
Add module into Android Studio application project:

- Android Studio way:
  (will copy almost all OpenCV Android SDK into your project, ~200Mb)

  Import module: Menu -> "File" -> "New" -> "Module" -> "Import Gradle project":
  Source directory: select this "sdk" directory
  Module name: ":opencv"

- or attach library module from OpenCV Android SDK
  (without copying into application project directory, allow to share the same module between projects)

  Edit "settings.gradle" and add these lines:

  def opencvsdk='<path_to_opencv_android_sdk_rootdir>'
  // You can put declaration above into gradle.properties file instead (including file in HOME directory),
  // but without 'def' and apostrophe symbols ('): opencvsdk=<path_to_opencv_android_sdk_rootdir>
  include ':opencv'
  project(':opencv').projectDir = new File(opencvsdk + '/sdk')
```

可以看出，这边有两种方法，android studio 的方法，会将 sdk 拷贝到工程中。第二种方法则没有拷贝，多个工程可以共用。这一步前面已经做了，这儿不用再做。

```
Add dependency into application module:

- Android Studio way:
  "Open Module Settings" (F4) -> "Dependencies" tab

- or add "project(':opencv')" dependency into app/build.gradle:

  dependencies {
      implementation fileTree(dir: 'libs', include: ['*.jar'])
      ...
      implementation project(':opencv')
  }
```

这一步前面也做了，采用 Android studio 方法。

```
Load OpenCV native library before using:

- avoid using of "OpenCVLoader.initAsync()" approach - it is deprecated
  It may load library with different version (from OpenCV Android Manager, which is installed separatelly on device)

- use "System.loadLibrary("opencv_java4")" or "OpenCVLoader.initDebug()"
  TODO: Add accurate API to load OpenCV native library
```

这一步是在 java 代码中。

```
Native C++ support (necessary to use OpenCV in native code of application only):

- Use find_package() in app/CMakeLists.txt:

  find_package(OpenCV 4.5 REQUIRED java)
  ...
  target_link_libraries(native-lib ${OpenCV_LIBRARIES})

- Add "OpenCV_DIR" and enable C++ exceptions/RTTI support via app/build.gradle
  Documentation about CMake options: https://developer.android.com/ndk/guides/cmake.html

  defaultConfig {
      ...
      externalNativeBuild {
          cmake {
              cppFlags "-std=c++11 -frtti -fexceptions"
              arguments "-DOpenCV_DIR=" + opencvsdk + "/sdk/native/jni" // , "-DANDROID_ARM_NEON=TRUE"
          }
      }
  }

- (optional) Limit/filter ABIs to build ('android' scope of 'app/build.gradle'):
  Useful information: https://developer.android.com/studio/build/gradle-tips.html (Configure separate APKs per ABI)

  splits {
      abi {
          enable true
          universalApk false
          reset()
          include 'armeabi-v7a' // , 'x86', 'x86_64', 'arm64-v8a'
      }
  }
```

可以看到有两个步骤是必须的，第三个步骤是可选的。

1. app/CMakeLists.txt
2. app/build.gradle

按上面做，没错。其中 externalNativeBuild 是在 defaultConfig 里

```
android {
    namespace 'com.example.cvapp'
    compileSdk 33

    defaultConfig {
        applicationId "com.example.cvapp"
        minSdk 21
        targetSdk 33
        versionCode 1
        versionName "1.0"

        testInstrumentationRunner "androidx.test.runner.AndroidJUnitRunner"

        externalNativeBuild {
            cmake {
                cppFlags "-std=c++11 -frtti -fexceptions"
                arguments "-DOpenCV_DIR=D:/library/OpenCV-android-sdk/sdk/native/jni" // , "-DANDROID_ARM_NEON=TRUE"
            }
        }
    }
```

### java 端

```java
    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();
        Log.i("cvapp", "onCameraFrame "+mRgba.width());

        FindFeatures(mGray.getNativeObjAddr(), mRgba.getNativeObjAddr());
        return mRgba;
    }

    public native void FindFeatures(long matAddrGr, long matAddrRgba);
```

一般是在 onCameraFrame 调用 c++ 函数。传送的 mat 的地址

### c++ 端

java 端写好后，会报错，因为 c++ 端还没相应函数。此时右键点击源码错误位置，可让AS自动生成相应C++函数。

jni 接口一般返回void，参数是mat 的地址，所以对这个mat的修改是会成呈现到java端。

```c++
extern "C"
JNIEXPORT void JNICALL
Java_com_example_cvapp_MainActivity_FindFeatures(JNIEnv *env, jobject thiz, jlong mat_addr_gr,
                                                 jlong mat_addr_rgba) {
    // TODO: implement FindFeatures()
    Mat& mGr  = *(Mat*)mat_addr_gr;
    Mat& mRgb = *(Mat*)mat_addr_rgba;
    
    // 直接修改 mRgb
```

### JNI 调用 c++类

[JNI调用C++类的方式](https://blog.csdn.net/xiaohan2909/article/details/50152997)

C++ 功能类和 JNI 可以做到分离，不用再创建 JNI 类（不知道有没有 JNI 类）。

可以建立一个 Java 类。也可以没有 java类，则相应的对象生成和销毁要放在 activity 中。

## 摄像头采集和问题解决

摄像头采集使用 opencv 的类。但会有一些问题。更灵活的解决方式要掌握底层的 Camera

1. [Camera API](https://developer.android.google.cn/guide/topics/media/camera?hl=zh-cn)
2. [android.hardware.camera2](https://developer.android.google.cn/reference/android/hardware/camera2/package-summary)

<img class="img-fluid" src="/images/2024-12-30-android-opencv-ndk/image-20230722141843054.png" alt="image-20230722141843054" style="zoom:80%;" />

### OpenCV之调用设备摄像头

根据 OpenCV-android-sdk\samples\face-detection 例程，总结如下

1. layout 中显示部分使用了 org.opencv.android.JavaCameraView 
2. activity 为 CameraActivity（注意是 org.opencv.android 的CameraActivity），同时实现了 CameraBridgeViewBase.CvCameraViewListener2 接口
3. **JavaCameraView继承自CameraBridgeViewBase，CameraBridgeViewBase又继承自SurfaceView**，JavaCameraView是可以显示摄像头捕获到的帧数据的View，CameraBridgeViewBase类中CvCameraViewListener2接口提供了摄像头onCameraViewStarted、onCameraViewStopped以及onCameraFrame回调。**我们要对摄像头捕获的每一帧数据进行操作就需要在OnCameraFrame回调中进行处理。**
4. CameraActivity 继承了 android.app.Activity

参考：

1. OpenCV-android-sdk\samples\face-detection 例程
2. [android中使用OpenCV之调用设备摄像头](https://www.jianshu.com/p/3462dac8a876)
3. [OpenCV4 Android 调用摄像头_opencv4调用手机摄像头](https://blog.csdn.net/chy555chy/article/details/106788660) 这一篇感觉有些东西，看看

>1. 首先，要继承 CameraActivity，之前已经说过了。这个基类会去申请权限，然后通知 javaCameraView 已获取到权限，可以正常使用。
>2. 复写父类的 getCameraViewList 方法，将 javaCameraView 回送回去，这样当权限已被赋予时，就可以通知到预览界面开始正常工作了。
>3. OpenCV 已经为我们实现了 Camera 和 Camera2 的函数，如果应用最低版本 minSdkVersion > 5.0，建议使用 JavaCamera2View 的相关函数，否则使用 JavaCameraView。
>4. 在 onResume 时判断 opencv 库是否加载完毕，然后启用预览视图。在 onPause 时由于界面被遮挡，此时应该暂停摄像头的预览以节省手机性能和电量损耗。
>5. 切换前后摄像头时，要先禁用，设置完后启用才会生效。
>6. Camera2 和 Camera 的绝大部分差异 OpenCV 均已经为我们屏蔽在类的内部了，唯一的差别就是两者实现的 CvCameraViewListener 监听器里的预览函数 onCameraFrame 的参数略有不同。从下面的源码可以看出 CvCameraViewListener2 的 inputFrame 由 Mat 类型改为了 CvCameraViewFrame 类型，它额外提供了一个转化为灰度图的接口。

### 图像采集时方向和大小问题

#### 选择主摄像头或前置

在 layout 文件中

#### 图像方向

opencv android 获取的图像是 landscape 的，我参照下面几个网页，做了个临时解决方案，这种解决方案是修改底层 CameraBridgeViewBase 代码，仅涉及到渲染时进行旋转和缩放，但是更底层的解决方案应该是控制 camera，以后再改善。

修改 opencv CameraBridgeViewBase 类的 deliverAndDrawFrame 方法

```java
    protected void deliverAndDrawFrame(CvCameraViewFrame frame) {
        Mat modified;

        if (mListener != null) {
            modified = mListener.onCameraFrame(frame);
        } else {
            modified = frame.rgba();
        }

        boolean bmpValid = true;
        if (modified != null) {
            try {
                Utils.matToBitmap(modified, mCacheBitmap);
            } catch(Exception e) {
                Log.e(TAG, "Mat type: " + modified);
                Log.e(TAG, "Bitmap type: " + mCacheBitmap.getWidth() + "*" + mCacheBitmap.getHeight());
                Log.e(TAG, "Utils.matToBitmap() throws an exception: " + e.getMessage());
                bmpValid = false;
            }
        }

        if (bmpValid && mCacheBitmap != null) {
            Canvas canvas = getHolder().lockCanvas();
            if (canvas != null) {
                canvas.drawColor(0, android.graphics.PorterDuff.Mode.CLEAR);
                if (BuildConfig.DEBUG)
                    Log.d(TAG, "mStretch value: " + mScale);

//                if (mScale != 0) {
//                    canvas.drawBitmap(mCacheBitmap, new Rect(0,0,mCacheBitmap.getWidth(), mCacheBitmap.getHeight()),
//                         new Rect((int)((canvas.getWidth() - mScale*mCacheBitmap.getWidth()) / 2),
//                         (int)((canvas.getHeight() - mScale*mCacheBitmap.getHeight()) / 2),
//                         (int)((canvas.getWidth() - mScale*mCacheBitmap.getWidth()) / 2 + mScale*mCacheBitmap.getWidth()),
//                         (int)((canvas.getHeight() - mScale*mCacheBitmap.getHeight()) / 2 + mScale*mCacheBitmap.getHeight())), null);
//                } else {
//                     canvas.drawBitmap(mCacheBitmap, new Rect(0,0,mCacheBitmap.getWidth(), mCacheBitmap.getHeight()),
//                         new Rect((canvas.getWidth() - mCacheBitmap.getWidth()) / 2,
//                         (canvas.getHeight() - mCacheBitmap.getHeight()) / 2,
//                         (canvas.getWidth() - mCacheBitmap.getWidth()) / 2 + mCacheBitmap.getWidth(),
//                         (canvas.getHeight() - mCacheBitmap.getHeight()) / 2 + mCacheBitmap.getHeight()), null);
//                }

                //  Start of the fix
                Matrix matrix = new Matrix();
                matrix.preTranslate( ( canvas.getWidth() - mCacheBitmap.getWidth() ) / 2f, ( canvas.getHeight() - mCacheBitmap.getHeight() ) / 2f );
                matrix.postRotate( -90f, canvas.getWidth() / 2f, canvas.getHeight() / 2f );
                float scale = (float) canvas.getWidth() / (float) mCacheBitmap.getHeight();
                matrix.postScale(scale, scale, canvas.getWidth() / 2f , canvas.getHeight() / 2f );
                canvas.drawBitmap( mCacheBitmap, matrix, null );
                // End of the fix

                if (mFpsMeter != null) {
                    mFpsMeter.measure();
                    mFpsMeter.draw(canvas, 20, 30);
                }
                getHolder().unlockCanvasAndPost(canvas);
            }
        }
    }
```

其中，注释部分是原来的， fix 之间是新增的。

##### TODO：

这边可能还有一个左右翻转的问题，需要再加一个 scale -1。后续有空再改

参考

1. 这个地方有个 [fix Rotate camera preview to Portrait Android OpenCV Camera](https://stackoverflow.com/questions/14816166/rotate-camera-preview-to-portrait-android-opencv-camera/57181157#57181157)
2. [android.graphics.Matrix](https://android.googlesource.com/platform/frameworks/base/+/75efd5d/graphics/java/android/graphics/Matrix.java) 说明
3. [Understanding Android Matrix transformations](https://medium.com/a-problem-like-maria/understanding-android-matrix-transformations-25e028f56dc7#id_token=eyJhbGciOiJSUzI1NiIsImtpZCI6ImEzYmRiZmRlZGUzYmFiYjI2NTFhZmNhMjY3OGRkZThjMGIzNWRmNzYiLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20iLCJuYmYiOjE2ODk5OTA2NzUsImF1ZCI6IjIxNjI5NjAzNTgzNC1rMWs2cWUwNjBzMnRwMmEyamFtNGxqZGNtczAwc3R0Zy5hcHBzLmdvb2dsZXVzZXJjb250ZW50LmNvbSIsInN1YiI6IjEwMTc1NjE1NTQ4MjA2NzgwMDM4MSIsImVtYWlsIjoia2Noc2p0dUBnbWFpbC5jb20iLCJlbWFpbF92ZXJpZmllZCI6dHJ1ZSwiYXpwIjoiMjE2Mjk2MDM1ODM0LWsxazZxZTA2MHMydHAyYTJqYW00bGpkY21zMDBzdHRnLmFwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29tIiwibmFtZSI6IktldmluIENoZW4iLCJwaWN0dXJlIjoiaHR0cHM6Ly9saDMuZ29vZ2xldXNlcmNvbnRlbnQuY29tL2EvQUFjSFR0ZndBRFgzVVB3VnpuMWFtcGVOc0EzU0h6elZBNmVLQkpfTnhOcHFhZEJaPXM5Ni1jIiwiZ2l2ZW5fbmFtZSI6IktldmluIiwiZmFtaWx5X25hbWUiOiJDaGVuIiwiaWF0IjoxNjg5OTkwOTc1LCJleHAiOjE2ODk5OTQ1NzUsImp0aSI6IjkxYmFiZGQ2YmM3ZjljMTFhMGJhYTZlZmMwNzc5MzZlMTdmZGU4OTMifQ.vHOtdorzRaOhZae6BB1ESbv58xklGFAwO6numO7Rg2XqY43Dnv7huhPon2ehhppGtsSQb69fKEGGr7QOepKhgQpW90ro7NCgGmRhKD5f9Pu8zcWO4XvuW9C0gTcw5vtj51oOMdTsaZcI6W50gCOd41K9GrqFGQuR1HYuG-2T23Cgjn-eP2TLDP2Lr3P6166FdrFy2w92trqCPEI4rFWaYU2FmlJxiBNML64mGyN6-vr2oZ_ofXTeRQB5je5YpCHMGFGNrSr6XALWLsJQAJI-RWKOcsKJ29Fy4KNpunLnTafDKLLgAz5VoIpLdIDmW3rrQ5mxYEr8OtYTA9wbwjyOhw)
3. [Android中使用Opencv自带JavaCameraView实现高帧率竖屏显示](https://blog.csdn.net/abcd1115313634/article/details/77982028)

#### 图像大小

默认是 800x600

## 调试

在安卓开发中，有多种方式可以保存调试信息，以下是几种常见的方法：

1. 使用Logcat：Logcat是Android Studio自带的日志工具，可以在开发过程中打印调试信息。你可以使用Log.d()、Log.e()等方法打印不同级别的日志信息，并在Logcat控制台中查看。你可以使用过滤选项来过滤出你需要查看的日志信息。如果你想保存Logcat的日志，可以将日志导出到文件中。
2. 使用SharedPreferences：SharedPreferences是Android提供的轻量级存储解决方案，可以用来保存应用程序的配置信息和状态信息。你可以使用SharedPreferences保存一些调试信息，如应用程序的版本号、一些标志位等。
3. 使用文件：你可以将调试信息写入到本地文件中，然后在需要的时候读取。你可以使用Java中的FileInputStream、FileOutputStream等类来操作文件。
4. 使用第三方库：有许多第三方库可以用来保存调试信息，如Bugly、Firebase Crashlytics等。这些库可以帮助你收集应用程序的崩溃日志和异常信息，并将其报告给你。

[使用 Logcat 写入和查看日志](https://developer.android.google.cn/studio/debug/am-logcat?hl=zh-cn#java)

## Android.app.Activity

[Activity 类](https://developer.android.google.cn/guide/components/activities/intro-activities?hl=zh-cn)是 Android 应用的关键组件，而 Activity 的启动和组合方式则是该平台应用模型的基本组成部分。在编程范式中，应用是通过 main() 方法启动的，而 Android 系统与此不同，它会调用与其生命周期特定阶段相对应的特定回调方法来启动 Activity 实例中的代码。

#### onCreate()

您必须实现此回调，它会在系统创建您的 Activity 时触发。您的实现应该初始化 Activity 的基本组件：例如，您的应用应该在此处创建视图并将数据绑定到列表。最重要的是，您必须在此处调用 `setContentView()` 来定义 Activity 界面的布局。

`onCreate()` 完成后，下一个回调将是 `onStart()`。

#### onStart()

`onCreate()` 退出后，Activity 将进入“已启动”状态，并对用户可见。此回调包含 Activity 进入前台与用户进行互动之前的最后准备工作。

#### onResume()

系统会在 Activity 开始与用户互动之前调用此回调。此时，该 Activity 位于 Activity 堆栈的顶部，并会捕获所有用户输入。**应用的大部分核心功能都是在 `onResume()` 方法中实现的。**

`onResume()` 回调后面总是跟着 `onPause()` 回调。

#### onPause()

当 Activity 失去焦点并进入“已暂停”状态时，系统就会调用 `onPause()`。例如，当用户点按“返回”或“最近使用的应用”按钮时，就会出现此状态。当系统为您的 Activity 调用 `onPause()` 时，从技术上来说，这意味着您的 Activity 仍然部分可见，但大多数情况下，这表明用户正在离开该 Activity，该 Activity 很快将进入“已停止”或“已恢复”状态。

如果用户希望界面继续更新，则处于“已暂停”状态的 Activity 也可以继续更新界面。例如，显示导航地图屏幕或播放媒体播放器的 Activity 就属于此类 Activity。即使此类 Activity 失去了焦点，用户仍希望其界面继续更新。

您**不应使用** `onPause()` 来保存应用或用户数据、进行网络呼叫或执行数据库事务。有关保存数据的信息，请参阅[保存和恢复 Activity 状态](https://developer.android.google.cn/guide/components/activities/activity-lifecycle?hl=zh-cn#saras)。

`onPause()` 执行完毕后，下一个回调为 `onStop()`或 `onResume()`，具体取决于 Activity 进入“已暂停”状态后发生的情况。

#### onStop()

当 Activity 对用户不再可见时，系统会调用 `onStop()`。出现这种情况的原因可能是 Activity 被销毁，新的 Activity 启动，或者现有的 Activity 正在进入“已恢复”状态并覆盖了已停止的 Activity。在所有这些情况下，停止的 Activity 都将完全不再可见。

系统调用的下一个回调将是 `onRestart()`（如果 Activity 重新与用户互动）或者 `onDestroy()`（如果 Activity 彻底终止）。

#### onRestart()

当处于“已停止”状态的 Activity 即将重启时，系统就会调用此回调。`onRestart()` 会从 Activity 停止时的状态恢复 Activity。

此回调后面总是跟着 `onStart()`。

#### onDestroy()

系统会在销毁 Activity 之前调用此回调。

此回调是 Activity 接收的最后一个回调。通常，实现 `onDestroy()` 是为了确保在销毁 Activity 或包含该 Activity 的进程时释放该 Activity 的所有资源。

## 其他问题

### 不显示 menu

AndroidManifest.xml 中

```
android:theme="@android:style/Theme.Holo.Light"
```

或换成其他theme

### 设定启动时的 activity

启动时的 activity 称为 main activity。可以有多个  main activity。这个时候好像是在 manifest 排第一位的最终胜出。

一般可以只保留一个  main activity。方法是把其他的 category 标签注释掉。

```xml
        <activity
            android:name=".FdActivity"
            android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />

<!--                <category android:name="android.intent.category.LAUNCHER" />-->
            </intent-filter>
        </activity>
```

### Asserts 目录

[Android资源文件分类](https://www.cnblogs.com/guanxinjing/p/9708583.html)

