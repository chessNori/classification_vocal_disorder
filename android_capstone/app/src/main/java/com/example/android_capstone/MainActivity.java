package com.example.android_capstone;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.annotation.SuppressLint;
import android.content.res.AssetFileDescriptor;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Environment;
import android.view.View;
import android.widget.Button;
import android.util.Log;
import android.widget.TextView;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.concurrent.atomic.AtomicBoolean;

import org.tensorflow.lite.Interpreter;

public class MainActivity extends AppCompatActivity {
    private static final int RECORDER_SAMPLE_RATE = 8000;
    private static final int CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO;
    private static final int AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT;
    int AUDIO_SOURCE = MediaRecorder.AudioSource.MIC;
    int BUFFER_SIZE_RECORDING = AudioRecord.getMinBufferSize(RECORDER_SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT);
    private final AtomicBoolean recordingInProgress = new AtomicBoolean(false);
    private AudioRecord audioRecord = null;
    private Thread recordingThread = null;

    private Button startButton;
    private Button stopButton;
    private Button resultButton;
    TextView progress;
    TextView result;
    TextView guide;

    float[][] res_wave = new float[8][66560];
    float[][] output = new float[8][4];
    float[] mean = new float[4];
    int res_argmax = 0;

    int raw = 0;
    int k = 0;

    private Interpreter getTfliteInterpreter(String modelPath) {
        try {
            return new Interpreter(loadModelFile(modelPath));
        }
        catch (Exception e) {
            e.printStackTrace();
        }
        return null;
    }

    public MappedByteBuffer loadModelFile(String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = MainActivity.this.getAssets().openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        guide = findViewById(R.id.guide);
        guide.setText("please record in the silent place");

        String[] permissions = {android.Manifest.permission.RECORD_AUDIO, android.Manifest.permission.WRITE_EXTERNAL_STORAGE, android.Manifest.permission.READ_EXTERNAL_STORAGE};
        ActivityCompat.requestPermissions(this, permissions, 0);

        startButton = (Button) findViewById(R.id.btnStart);
        stopButton = (Button) findViewById(R.id.btnStop);
        resultButton = (Button) findViewById(R.id.btnResult);

        startButton.setEnabled(true);
        stopButton.setEnabled(false);
        resultButton.setEnabled(false);

        startButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                startRecording();

                startButton.setEnabled(false);
                stopButton.setEnabled(true);
                resultButton.setEnabled(false);

            }
        });

        stopButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                progress = findViewById(R.id.progress);
                stopButton.setEnabled(false);

                stopRecording();

                if(raw == 7){
                    startButton.setEnabled(false);
                    resultButton.setEnabled(true);
                } else {
                    startButton.setEnabled(true);
                    resultButton.setEnabled((false));
                }

                stopButton.setEnabled(false);
                progress.setText("진행도: "+ Integer.toString(raw + 1));
            }
        });

        resultButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                result = findViewById(R.id.result);
                resultButton.setEnabled(false);
                startButton.setEnabled(true);

                Log.d("res_argmax", Integer.toString(res_argmax));

                if (res_argmax == 0) {
                    result.setText("병명: Normal");
                } else if (res_argmax == 1) {
                    result.setText("병명: Papilloma");
                } else if (res_argmax == 2) {
                    result.setText("병명: Paralysis");
                } else {
                    result.setText("병명: Vox Senilis");
                }
                raw = 0;
                for(int a =0;a<8;a++){
                    for(int b=0;b<66560;b++){
                        res_wave[a][b] = 0.0F;
                    }
                }
            }
        });
    }

    @SuppressLint("MissingPermission")
    private void startRecording() {
        audioRecord = new AudioRecord(AUDIO_SOURCE, RECORDER_SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT, BUFFER_SIZE_RECORDING);
        audioRecord.startRecording();
        recordingInProgress.set(true);
        recordingThread = new Thread(new MainActivity.RecordingRunnable(), "Recording Thread");
        recordingThread.start();
    }

    private void stopRecording() {
        if (null == audioRecord) {
            return;
        }

        recordingInProgress.set(false);
        audioRecord.stop();
        audioRecord.release();
        audioRecord = null;
        recordingThread = null;
    }

    private class RecordingRunnable implements Runnable {
        @Override
        public void run() {
            final String folderName = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS).getAbsolutePath() + "/STTFile";
            String name = "test.raw";

            File dir = new File(folderName);
            dir.mkdirs();
            final File file = new File(dir, name);

            final ByteBuffer readData = ByteBuffer.allocateDirect(BUFFER_SIZE_RECORDING);

            FileOutputStream outStream = null;
            try {
                outStream = new FileOutputStream(file);
            } catch (FileNotFoundException e) {
                throw new RuntimeException(e);
            }

            while (recordingInProgress.get()) {
                int result = audioRecord.read(readData, BUFFER_SIZE_RECORDING);
                if (result < 0) {
                    throw new RuntimeException("Reading of audio buffer failed: " +
                            getBufferReadFailureReason(result));
                }

                try{
                    outStream.write(readData.array(), 0, BUFFER_SIZE_RECORDING);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
            try{
                outStream.close();}
            catch (IOException e){
                throw new RuntimeException(e);
            }

            FileInputStream fis = null;
            int upward;
            int backward;

            try {
                fis = new FileInputStream(file);
                for(int i = 0; i < 66560; i++) {
                    upward = fis.read();
                    if(upward == -1) {
                        break;
                    }
                    backward = fis.read();
                    res_wave[raw][i] = (float)((short)(upward | (backward << 8)));
                }
                fis.close();
            } catch(IOException e) {
                throw new RuntimeException(e);
            }

            Log.d("file_input1001_" + raw, Float.toString(res_wave[raw][8001]));
            Log.d("file_input1002_" + raw, Float.toString(res_wave[raw][8002]));
            Log.d("file_input1003_" + raw, Float.toString(res_wave[raw][8003]));
            Log.d("file_input1004_" + raw, Float.toString(res_wave[raw][8004]));
            Log.d("file_input1005_" + raw, Float.toString(res_wave[raw][8005]));
            Log.d("file_input1006_" + raw, Float.toString(res_wave[raw][8006]));
            Log.d("file_input1007_" + raw, Float.toString(res_wave[raw][8007]));
            Log.d("file_input1008_" + raw, Float.toString(res_wave[raw][8008]));

            raw += 1;

            if (raw == 8) {
                int argmax = 0;

                Interpreter lite = getTfliteInterpreter("yhs_full_model.tflite");
                lite.run(res_wave, output);

                for (int j = 0; j < 8; j++) {
                    Log.d("Batch Results", Float.toString(output[j][0]) + Float.toString(output[j][1]) + Float.toString(output[j][2]) + Float.toString(output[j][3]));
                }
                for (k = 0; k < 4; k++) {
                    float sum = 0;
                    for (int l = 0; l < 8; l++) {
                        sum += output[l][k];
                    }
                    mean[k] = sum;
                    if (mean[argmax] < mean[k]) {
                        argmax = k;
                    }
                }
                res_argmax = argmax;
                Log.d("Probability1", Float.toString(mean[0]));
                Log.d("Probability2", Float.toString(mean[1]));
                Log.d("Probability3", Float.toString(mean[2]));
                Log.d("Probability4", Float.toString(mean[3]));
                Log.d("Model Result", Integer.toString(argmax));
            }
        }
    }
    private String getBufferReadFailureReason(int errorCode) {
        switch (errorCode) {
            case AudioRecord.ERROR_INVALID_OPERATION:
                return "ERROR_INVALID_OPERATION";
            case AudioRecord.ERROR_BAD_VALUE:
                return "ERROR_BAD_VALUE";
            case AudioRecord.ERROR_DEAD_OBJECT:
                return "ERROR_DEAD_OBJECT";
            case AudioRecord.ERROR:
                return "ERROR";
            default:
                return "Unknown (" + errorCode + ")";
        }
    }
}