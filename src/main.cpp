#include "mbed.h"
#include "ei_run_classifier.h"
#include "numpy.hpp"

#include "LSM303AGRAccSensor.h"

#include "USBSerial.h"
USBSerial ser;

SPI devSPI(PB_15, NC, PB_13);  // 3-wires SPI on SensorTile  
static LSM303AGRAccSensor accelerometer(&devSPI, PC_4);

DigitalOut led((PinName)0x6C);

volatile bool trig = 0;
Ticker timer;
uint8_t id;

static float features[177]; /* = {
    // copy raw features here (for example from the 'Live classification' page)
    -744, -233, -627, -802, -268, -631, -861, -256, -596, -873, -225, -529, -904, -194, -557, -873, -178, -533, -841, -241, -522, -884, -229, -537, -904, -178, -487, -927, -186, -483, -947, -112, -420, -939, -85, -436, -857, -108, -498, -857, -194, -545, -865, -229, -479, -892, -202, -475, -916, -124, -502, -916, -116, -487, -912, -143, -459, -927, -175, -424, -873, -178, -440, -841, -175, -506, -818, -210, -553, -826, -237, -553, -830, -198, -553, -857, -171, -557, -931, -151, -518, -1013, -89, -459, -1017, -128, -448, -900, -186, -568, -916, -163, -490, -919, -175, -401, -931, -202, -444, -904, -331, -494, -923, -416, -459, -951, -206, -529, -935, -132, -721, -974, -206, -658, -986, 23, -740, -802, 398, -553, -1005, -61, -214, -935, -69, -303, -1064, -7, -241, -947, -108, -264, -982, -288, -362, -955, -19, -432, -982, -256, -526, -1048, -260, -124, -1048, -143, -38, -1060, -143, -38, -884, -136, -451, -783, -73, -701, -713, -128, -763, -701, -342, -756, -662, -537, -775, -650, -448, -682, -724, -385, -724, -1212, -136, -206, -900, -432, -514, -880, -108, -346, -970, -120, -327, -877, -42, -346, -896, -151, -514, -752, -194, -502, -791, -377, -623, -884, -444, -604, -986, -397, -799, -1001, -358, -802, -619, -225, -370, -545, -202, -451, -1196, -108, -455, -830, -81, -424, -1005, 23, -545, -1056, -11, -580, -1036, 59, -381, -951, 43, -194, -966, 20, -178, -1033, 121, -393, -1099, 234, -689, -1302, 31, -334, -1040, -373, -171, -810, -553, -494, -635, -334, -658, -682, -97, -736, -717, -171, -763, -670, -299, -748, -689, -362, -697, -643, -366, -705, -693, -342, -721, -670, -315, -658, -744, -268, -635, -838, -237, -584, -916, -175, -479, -970, -147, -463, -1001, -120, -420, -955, -132, -385, -927, -147, -416, -939, -69, -393, -939, -93, -451
    // see https://docs.edgeimpulse.com/docs/running-your-impulse-mbed
};*/

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
    timer.attach_us(&sample,10000);
    
    while (1) {
        led = 0;
        int nsamples = 0;
        while(nsamples<177){
            
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
            ThisThread::sleep_for(100);
        } else if (out == 1) {
            led = 1;
            ThisThread::sleep_for(100);
            led = 0;
            ThisThread::sleep_for(100);
            led = 1;
            ThisThread::sleep_for(100);
        }

        //ThisThread::sleep_for(2000);
    }
}
