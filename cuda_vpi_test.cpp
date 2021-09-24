#include <opencv2/core/version.hpp>
 #if CV_MAJOR_VERSION >= 3
 #    include <opencv2/imgcodecs.hpp>
 #    include <opencv2/videoio.hpp>
 #else
 #    include <opencv2/highgui/highgui.hpp>
 #endif
  
 #include <opencv2/imgproc/imgproc.hpp>
 #include <vpi/OpenCVInterop.hpp>
  
 #include <vpi/Image.h>
 #include <vpi/Status.h>
 #include <vpi/Stream.h>
 #include <vpi/algo/ConvertImageFormat.h>
 #include <vpi/algo/TemporalNoiseReduction.h>
  
 #include <algorithm>
 #include <cstring> // for memset
 #include <fstream>
 #include <iostream>
 #include <map>
 #include <sstream>
 #include <vector>
  
 #define CHECK_STATUS(STMT)                                    \
     do                                                        \
     {                                                         \
         VPIStatus status = (STMT);                            \
         if (status != VPI_SUCCESS)                            \
         {                                                     \
             char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];       \
             vpiGetLastStatusMessage(buffer, sizeof(buffer));  \
             std::ostringstream ss;                            \
             ss << vpiStatusGetName(status) << ": " << buffer; \
             throw std::runtime_error(ss.str());               \
         }                                                     \
     } while (0);
  
 int main(int argc, char *argv[])
 {
     // OpenCV image that will be wrapped by a VPIImage.
     // Define it here so that it's destroyed *after* wrapper is destroyed
     cv::Mat cvFrame;
  
     // Declare all VPI objects we'll need here so that we
     // can destroy them at the end.
     VPIStream stream     = NULL;
     VPIImage imgPrevious = NULL, imgCurrent = NULL, imgOutput = NULL;
     VPIImage frameBGR = NULL;
     VPIPayload tnr    = NULL;
  
     // main return value
     int retval = 0;
  
     try
     {
         // =============================
         // Parse command line parameters
  
         if (argc != 3)
         {
             throw std::runtime_error(std::string("Usage: ") + argv[0] + " <vic|cuda> <input_video>");
         }
  
         std::string strBackend    = argv[1];
         std::string strInputVideo = argv[2];
  
         // Now parse the backend
         VPIBackend backend;
  
         if (strBackend == "cuda")
         {
             backend = VPI_BACKEND_CUDA;
         }
         else if (strBackend == "vic")
         {
             backend = VPI_BACKEND_VIC;
         }
         else
         {
             throw std::runtime_error("Backend '" + strBackend + "' not recognized, it must be either cuda or vic.");
         }
  
         // ===============================
         // Prepare input and output videos
  
         // Load the input video
         cv::VideoCapture invid;
         if (!invid.open(strInputVideo))
         {
             throw std::runtime_error("Can't open '" + strInputVideo + "'");
         }
  
         // Open the output video for writing using input's characteristics
 #if CV_MAJOR_VERSION >= 3
         int w                      = invid.get(cv::CAP_PROP_FRAME_WIDTH);
         int h                      = invid.get(cv::CAP_PROP_FRAME_HEIGHT);
         int fourcc                 = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
         double fps                 = invid.get(cv::CAP_PROP_FPS);
         std::string extOutputVideo = ".mp4";
 #else
         // MP4 support with OpenCV-2.4 has issues, we'll use
         // avi/mpeg instead.
         int w                      = invid.get(CV_CAP_PROP_FRAME_WIDTH);
         int h                      = invid.get(CV_CAP_PROP_FRAME_HEIGHT);
         int fourcc                 = CV_FOURCC('M', 'P', 'E', 'G');
         double fps                 = invid.get(CV_CAP_PROP_FPS);
         std::string extOutputVideo = ".avi";
 #endif
  
         // Create the output video
         cv::VideoWriter outVideo("denoised_" + strBackend + extOutputVideo, fourcc, fps, cv::Size(w, h));
         if (!outVideo.isOpened())
         {
             throw std::runtime_error("Can't create output video");
         }
  
         // =================================
         // Allocate all VPI resources needed
  
         // We'll use the backend passed to run remap algorithm and the CUDA to do image format
         // conversions, therefore we have to force enabling of CUDA backend, along with the
         // desired backend.
         CHECK_STATUS(vpiStreamCreate(VPI_BACKEND_CUDA | backend, &stream));
  
         CHECK_STATUS(vpiImageCreate(w, h, VPI_IMAGE_FORMAT_NV12_ER, 0, &imgPrevious));
         CHECK_STATUS(vpiImageCreate(w, h, VPI_IMAGE_FORMAT_NV12_ER, 0, &imgCurrent));
         CHECK_STATUS(vpiImageCreate(w, h, VPI_IMAGE_FORMAT_NV12_ER, 0, &imgOutput));
  
         // Create a Temporal Noise Reduction payload configured to process NV12_ER
         // frames under indoor medium light
         CHECK_STATUS(vpiCreateTemporalNoiseReduction(backend, w, h, VPI_IMAGE_FORMAT_NV12_ER, VPI_TNR_DEFAULT,
                                                      VPI_TNR_PRESET_INDOOR_MEDIUM_LIGHT, 1, &tnr));
  
         // ====================
         // Main processing loop
  
         int curFrame = 0;
         while (invid.read(cvFrame))
         {
             printf("Frame: %d\n", ++curFrame);
  
             // frameBGR isn't allocated yet?
             if (frameBGR == NULL)
             {
                 // Create a VPIImage that wraps the frame
                 CHECK_STATUS(vpiImageCreateOpenCVMatWrapper(cvFrame, 0, &frameBGR));
             }
             else
             {
                 // reuse existing VPIImage wrapper to wrap the new frame.
                 CHECK_STATUS(vpiImageSetWrappedOpenCVMat(frameBGR, cvFrame));
             }
  
             // First convert it to NV12_ER
             CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, frameBGR, imgCurrent, NULL));
  
             // Apply temporal noise reduction
             // For first frame, we have to pass NULL as previous frame,
             // this will reset internal state.
             CHECK_STATUS(vpiSubmitTemporalNoiseReduction(stream, 0, tnr, curFrame == 1 ? NULL : imgPrevious, imgCurrent,
                                                          imgOutput));
  
             // Convert output back to BGR
             CHECK_STATUS(vpiSubmitConvertImageFormat(stream, VPI_BACKEND_CUDA, imgOutput, frameBGR, NULL));
             CHECK_STATUS(vpiStreamSync(stream));
  
             // Now add it to the output video stream
             VPIImageData imgdata;
             CHECK_STATUS(vpiImageLock(frameBGR, VPI_LOCK_READ, &imgdata));
  
             cv::Mat outFrame;
             CHECK_STATUS(vpiImageDataExportOpenCVMat(imgdata, &outFrame));
             outVideo << outFrame;
  
             CHECK_STATUS(vpiImageUnlock(frameBGR));
  
             // this iteration's output will be next's previous. Previous, which would be discarded, will be reused
             // to store next frame.
             std::swap(imgPrevious, imgOutput);
         };
     }
     catch (std::exception &e)
     {
         std::cerr << e.what() << std::endl;
         retval = 1;
     }
  
     // =========================
     // Destroy all VPI resources
     vpiStreamDestroy(stream);
     vpiPayloadDestroy(tnr);
     vpiImageDestroy(imgPrevious);
     vpiImageDestroy(imgCurrent);
     vpiImageDestroy(imgOutput);
     vpiImageDestroy(frameBGR);
  
     return retval;
 }
  
 // vim: ts=8:sw=4:sts=4:et:ai
