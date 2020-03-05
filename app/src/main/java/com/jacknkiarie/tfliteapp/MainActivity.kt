package com.jacknkiarie.tfliteapp

import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.nio.MappedByteBuffer


class MainActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialization code
// Create an ImageProcessor with all ops required. For more ops, please
// refer to the ImageProcessor Architecture section in this README.
        // Initialization code
// Create an ImageProcessor with all ops required. For more ops, please
// refer to the ImageProcessor Architecture section in this README.
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
            .build()

// Create a TensorImage object, this creates the tensor the TensorFlow Lite
// interpreter needs
        // Create a TensorImage object, this creates the tensor the TensorFlow Lite
// interpreter needs
        var tImage = TensorImage(DataType.UINT8)

// Analysis code for every frame
// Preprocess the image
        // Analysis code for every frame
// Preprocess the image
        val bitmap = BitmapFactory.decodeResource(resources, R.drawable.leaf)
        tImage.load(bitmap)
        tImage = imageProcessor.process(tImage)

        // Create a container for the result and specify that this is a quantized model.
// Hence, the 'DataType' is defined as UINT8 (8-bit unsigned integer)
        // Create a container for the result and specify that this is a quantized model.
// Hence, the 'DataType' is defined as UINT8 (8-bit unsigned integer)
        val probabilityBuffer =
            TensorBuffer.createFixedSize(intArrayOf(1, 1001), DataType.UINT8)

        // Initialise the model
        try{
            var tfliteModel = FileUtil.loadMappedFile(this,
                "mobilenet_v1_0.25_128_quant.tflite")
            var tflite = Interpreter(tfliteModel)
            // Running inference
            if(tflite != null) {
                tflite.run(tImage.getBuffer(), probabilityBuffer.buffer)
                print("The result of model is: " + probabilityBuffer.getFloatArray())
            }
        } catch (e: Exception){
            Log.e("tfliteSupport", "Error reading model", e)
        }
    }
}
