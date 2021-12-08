#include "mbed.h"
#include "ei_run_classifier.h"
#include "ei_run_dsp.h"

#include "LSM303AGRAccSensor.h"

#include "USBSerial.h"
USBSerial ser;

SPI devSPI(PB_15, NC, PB_13);  // 3-wires SPI on SensorTile  
static LSM303AGRAccSensor accelerometer(&devSPI, PC_4);

DigitalOut led((PinName)0x6C);

volatile bool trig = 0;
Ticker timer;
uint8_t id;

static float features[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];

int raw_feature_get_data(size_t offset, size_t length, float *out_ptr) {
    memcpy(out_ptr, features + offset, length * sizeof(float));
    return 0;
}

void sample(){
    trig=1;
}

int main() {
    led = 1;
    int32_t axes[3];
    
    ser.printf("Edge Impulse standalone inferencing (Mbed)\n");
    accelerometer.init(NULL);
    accelerometer.enable();
    accelerometer.read_id(&id);
    ser.printf("LSM303AGR accelerometer           = 0x%X\r\n", id); 
    accelerometer.get_x_axes(axes);
    ser.printf("LSM303AGR [acc/mg]:      %6ld, %6ld, %6ld\r\n", axes[0], axes[1], axes[2]);
    
    timer.attach_us(&sample,int(EI_CLASSIFIER_INTERVAL_MS*1000));

    ser.printf("nn input frame size: %d\n",EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    
    while (1) {
        led = 0;
        int nsamples = 0;
        while(nsamples<EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE){
            
            while (!trig){}
            trig = 0;
            accelerometer.get_x_axes(axes);
            features[nsamples] = (float)axes[0];
            features[nsamples+1] = (float)axes[1];
            features[nsamples+2] = (float)axes[2];
            nsamples+=3;
        }
        
        
        if (sizeof(features) / sizeof(float) != EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
            ser.printf("The size of your 'features' array is not correct. Expected %d items, but had %u\n",
                EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, sizeof(features) / sizeof(float));
            return 1;
        }

        ei_impulse_result_t result = { 0 };
        
        // the features are stored into flash, and we don't want to load everything into RAM
        signal_t features_signal;
        features_signal.total_length = sizeof(features) / sizeof(features[0]);
        features_signal.get_data = &raw_feature_get_data;
        
        /* Reformat this with properties from dsp_blocks.h file
        
        matrix_t raw_f_ex = matrix_t(ei_dsp_config_40_axes_size,ei_dsp_config_40_axes_size);
        int ei_dsp = extract_raw_features(&features_signal, &raw_f_ex, &ei_dsp_config_40, EI_CLASSIFIER_FREQUENCY);
        ser.printf("raw feature extraction returned: %d\n", ei_dsp);

        matrix_t flatten_f_ex = matrix_t(7,ei_dsp_config_60_axes_size);
        ei_dsp = extract_flatten_features(&features_signal, &flatten_f_ex, &ei_dsp_config_60, EI_CLASSIFIER_FREQUENCY);
        ser.printf("flatten feature extraction returned: %d\n", ei_dsp);
        */
        
        // invoke the impulse
        EI_IMPULSE_ERROR res = run_classifier(&features_signal, &result, true);
        ser.printf("run_classifier returned: %d\n", res);

        if (res != 0) return 1;

        ser.printf("Predictions (DSP: %d ms., Classification: %d ms., Anomaly: %d ms.): \n",
            result.timing.dsp, result.timing.classification, result.timing.anomaly);
        
        int out = 2;
        // print the predictions
        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            ser.printf("%s: %.5f", result.classification[ix].label, result.classification[ix].value);
#if EI_CLASSIFIER_HAS_ANOMALY == 1
            ser.printf(", ");
#else
            if (ix != EI_CLASSIFIER_LABEL_COUNT - 1) {
                ser.printf("\n");
            }
            if (result.classification[ix].value > 0.9) out = ix;
                
#endif
        }
#if EI_CLASSIFIER_HAS_ANOMALY == 1
        ser.printf("%.3f", result.anomaly);
#endif
        ser.printf("\n");
        
        if (out == 0) {
            led = 1;
            ThisThread::sleep_for(100ms);
        } else if (out == 1) {
            led = 1;
            ThisThread::sleep_for(100ms);
            led = 0;
            ThisThread::sleep_for(100ms);
            led = 1;
            ThisThread::sleep_for(100ms);
        }

        //ThisThread::sleep_for(2000);
    }
}
