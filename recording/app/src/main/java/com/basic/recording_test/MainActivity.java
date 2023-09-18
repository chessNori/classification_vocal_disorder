package com.basic.recording_test;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
//import androidx.room.jarjarred.org.stringtemplate.v4.Interpreter;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
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
import org.w3c.dom.Text;

public class MainActivity extends AppCompatActivity {
//// raw pcm data (16kHz Sampling rate, 16 bit short-int, little-endian)
// 헤더 : RAW(header-less), 인코딩 : Signed 16-bit PCM

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


    // 16kHz Sampling rate
    private static final int RECORDER_SAMPLE_RATE = 8000;

    // 오디오 채널 MONO
    private static final int CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO;

    private static final int AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT;

    // 마이크에서 음성 받아온다.
    int AUDIO_SOURCE = MediaRecorder.AudioSource.MIC;

    // 사용할 버퍼 사이즈
    int BUFFER_SIZE_RECORDING = AudioRecord.getMinBufferSize(RECORDER_SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT);

    /**
     * Signals whether a recording is in progress (true) or not (false).
     * Boolean 타입 과 거의 비슷한 듯. 여러 쓰레드에 안전하다는 장점이 있는?
     */
    private final AtomicBoolean recordingInProgress = new AtomicBoolean(false);

    // 음성을 녹음하는 객체. 음성을 디지털 데이터로 변환하는.
    private AudioRecord audioRecord = null;

    // 일반 스레드, 데이터를 계속 받아와서 파일에 저장하는
    private Thread recordingThread = null;

    private Button startButton;
    private Button stopButton;
    private Button resultButton;

    int[][] test_wave = new int[8][66560];
    float[][] res_wave = new float[8][66560];
    float[][] output = new float[8][4];
    int raw = 0;
    int k = 0;
    float[] mean = new float[4];
    int res_argmax = 0;

    TextView progress;
    TextView result;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // 권한 요청.
        String[] permissions = {android.Manifest.permission.RECORD_AUDIO, android.Manifest.permission.WRITE_EXTERNAL_STORAGE, android.Manifest.permission.READ_EXTERNAL_STORAGE};
        ActivityCompat.requestPermissions(this, permissions, 0);

        startButton = (Button) findViewById(R.id.btnStart);
        startButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                // 녹음 시작
                startRecording();

                startButton.setEnabled(false);
                stopButton.setEnabled(true);
                resultButton.setEnabled(false);

            }
        });

        stopButton = (Button) findViewById(R.id.btnStop);
        stopButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                progress = findViewById(R.id.progress);
                stopButton.setEnabled(false);

                // 녹음 중지
                stopRecording();
                Log.d("raw_out", Integer.toString(raw));
                if(raw == 7){
                    startButton.setEnabled(false);
                    resultButton.setEnabled(true);
                }
                else {
                    startButton.setEnabled(true);
                    resultButton.setEnabled((false));
                }
                stopButton.setEnabled(false);

                progress.setText("진행도: "+ Integer.toString(raw + 1));
//                Log.d("Check raw var", Integer.toString(raw));

//                String a = Float.toString(res_wave[raw][50]) + " ";
//                for(int k = 1; k < 15; k++){
//                    a += Float.toString(res_wave[raw][50 + k]) + " ";
//                }
//
//              progress.setText("진행도: "+ Integer.toString(raw));
//                if (raw == 8){
//                    Log.d("외부결과", Integer.toString(res_argmax));
//                    raw = 0;  // reset raw
//                }
            }
        });

        resultButton = (Button) findViewById(R.id.btnResult);
        resultButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                result = findViewById(R.id.result);
                resultButton.setEnabled(false);

                startButton.setEnabled(true);
//                stopButton.setEnabled(false);
                Log.d("Check raw var", Integer.toString(raw));

                Log.d("외부결과", Integer.toString(res_argmax));

                /*String debug_text = "";
                for (int m = 0; m < 4; m++){
                    debug_text += Float.toString(mean[m]);
                    debug_text += "/";
                }
                result.setText(debug_text);  // Debug mode */

                if (res_argmax == 0){
                    result.setText("병명: Normal");
                }
                else if(res_argmax == 1){
                    result.setText("병명: Papilloma");
                }
                else if(res_argmax == 2){
                    result.setText("병명: Paralysis");
                }
                else{
                    result.setText("병명: Vox Senilis");
                }

                raw = 0;
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
            int i = 0;

            final String foldername = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS).getAbsolutePath() + "/STTFile";
            String name = "test.raw";

            File dir = new File(foldername);
            dir.mkdirs();
            final File file = new File(dir, name);

            // 음성 데이터 잠시 담아둘 버퍼 생성.
//            final ByteBuffer buffer = ByteBuffer.allocateDirect(BUFFER_SIZE_RECORDING);
            final ByteBuffer readData = ByteBuffer.allocateDirect(BUFFER_SIZE_RECORDING);

            FileOutputStream outStream = null;
            try {
                outStream = new FileOutputStream(file);
            } catch (FileNotFoundException e) {
                throw new RuntimeException(e);
            }

            // 녹음하는 동안 {} 안의 코드 실행.
            while (recordingInProgress.get()) {
                // audioRecord 객체에서 음성 데이터 가져옴.
                int result = audioRecord.read(readData, BUFFER_SIZE_RECORDING);
                if (result < 0) {
                    throw new RuntimeException("Reading of audio buffer failed: " +
                            getBufferReadFailureReason(result));
                }
//                res_wave[raw][i] = (float) result;
//                Log.d("raw_test", Integer.toString((raw)));
                i += 1;

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

            try {
                fis = new FileInputStream(file);
            } catch (FileNotFoundException e) {
                throw new RuntimeException(e);
            }
            Log.d("raw_in", Integer.toString(raw));

            try (DataInputStream dis = new DataInputStream(fis)) {
                for (int l = 0; l < 66560; l++) {
                    byte upward = dis.readByte();
                    byte backward = dis.readByte();
                    int fuck = (int)upward | (((int)backward << 8));
                    res_wave[raw][l] = (float)fuck;
                }
                 dis.close();
            } catch (IOException e) {
                e.printStackTrace();
            }

            raw += 1;

            Log.d("분석" + raw, Float.toString(res_wave[7][700]));
            Log.d("분석" + raw, Float.toString(res_wave[7][701]));
            Log.d("분석" + raw, Float.toString(res_wave[7][702]));
            Log.d("분석" + raw, Float.toString(res_wave[7][703]));
            Log.d("분석" + raw, Float.toString(res_wave[7][704]));
            Log.d("분석" + raw, Float.toString(res_wave[7][705]));
            Log.d("분석" + raw, Float.toString(res_wave[7][706]));
            Log.d("분석" + raw, Float.toString(res_wave[7][707]));

            if (raw == 8) {
                int argmax = 0;

                Interpreter lite = getTfliteInterpreter("yhs_full_model.tflite");
                lite.run(res_wave, output);

                for (int j = 0; j < 8; j++) {
                    Log.d("결과", Float.toString(output[j][0]) + Float.toString(output[j][1]) + Float.toString(output[j][2]) + Float.toString(output[j][3]));
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
                Log.d("최종결과", Integer.toString(argmax));

                Log.d("결과1", Float.toString(mean[0]));
                Log.d("결과2", Float.toString(mean[1]));
                Log.d("결과3", Float.toString(mean[2]));
                Log.d("결과4", Float.toString(mean[3]));
            }

//            buffer.clear();
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