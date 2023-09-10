using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Runtime.InteropServices;
using OpenCvSharp;
using OpenCvSharp.XImgProc;
using OpenCvSharp.XFeatures2D;

#if ENABLE_WINMD_SUPPORT
using HL2UnityPlugin;
#endif

public class ResearchModeVideoStream : MonoBehaviour
{
#if ENABLE_WINMD_SUPPORT
    HL2ResearchMode researchMode;
#endif

    enum DepthSensorMode
    {
        ShortThrow,
        LongThrow,
        None
    };
    [SerializeField] DepthSensorMode depthSensorMode = DepthSensorMode.ShortThrow;
    [SerializeField] bool enablePointCloud = true;

    TCPClient tcpClient;

    public GameObject depthPreviewPlane = null;
    private Material depthMediaMaterial = null;
    private Texture2D depthMediaTexture = null;
    private byte[] depthFrameData = null;

    public GameObject shortAbImagePreviewPlane = null;
    private Material shortAbImageMediaMaterial = null;
    private Texture2D shortAbImageMediaTexture = null;
    private byte[] shortAbImageFrameData = null;

    public GameObject longDepthPreviewPlane = null;
    private Material longDepthMediaMaterial = null;
    private Texture2D longDepthMediaTexture = null;
    private byte[] longDepthFrameData = null;

    public GameObject longAbImagePreviewPlane = null;
    private Material longAbImageMediaMaterial = null;
    private Texture2D longAbImageMediaTexture = null;
    private byte[] longAbImageFrameData = null;

    public GameObject LFPreviewPlane = null;
    private Material LFMediaMaterial = null;
    private Texture2D LFMediaTexture = null;
    private byte[] LFFrameData = null;

    public GameObject RFPreviewPlane = null;
    private Material RFMediaMaterial = null;
    private Texture2D RFMediaTexture = null;
    private byte[] RFFrameData = null;

    public UnityEngine.UI.Text text;

    public GameObject pointCloudRendererGo;
    public Color pointColor = Color.white;
    private PointCloudRenderer pointCloudRenderer;
    private float[] uvMap;

#if ENABLE_WINMD_SUPPORT
    Windows.Perception.Spatial.SpatialCoordinateSystem unityWorldOrigin;
#endif

    private void Awake()
    {
#if ENABLE_WINMD_SUPPORT
#if UNITY_2020_1_OR_NEWER // note: Unity 2021.2 and later not supported
        IntPtr WorldOriginPtr = UnityEngine.XR.WindowsMR.WindowsMREnvironment.OriginSpatialCoordinateSystem;
        unityWorldOrigin = Marshal.GetObjectForIUnknown(WorldOriginPtr) as Windows.Perception.Spatial.SpatialCoordinateSystem;
        //unityWorldOrigin = Windows.Perception.Spatial.SpatialLocator.GetDefault().CreateStationaryFrameOfReferenceAtCurrentLocation().CoordinateSystem;
#else
        IntPtr WorldOriginPtr = UnityEngine.XR.WSA.WorldManager.GetNativeISpatialCoordinateSystemPtr();
        unityWorldOrigin = Marshal.GetObjectForIUnknown(WorldOriginPtr) as Windows.Perception.Spatial.SpatialCoordinateSystem;
#endif
#endif
    }
    void Start()
    {
        if (depthSensorMode == DepthSensorMode.ShortThrow)
        {
            if (depthPreviewPlane != null)
            {
                depthMediaMaterial = depthPreviewPlane.GetComponent<MeshRenderer>().material;
                depthMediaTexture = new Texture2D(512, 512, TextureFormat.Alpha8, false);
                depthMediaMaterial.mainTexture = depthMediaTexture;
            }

            if (shortAbImagePreviewPlane != null)
            {
                shortAbImageMediaMaterial = shortAbImagePreviewPlane.GetComponent<MeshRenderer>().material;
                shortAbImageMediaTexture = new Texture2D(512, 512, TextureFormat.Alpha8, false);
                shortAbImageMediaMaterial.mainTexture = shortAbImageMediaTexture;
            }
            longDepthPreviewPlane.SetActive(false);
            longAbImagePreviewPlane.SetActive(false);
        }
        
        if (depthSensorMode == DepthSensorMode.LongThrow)
        {
            if (longDepthPreviewPlane != null)
            {
                longDepthMediaMaterial = longDepthPreviewPlane.GetComponent<MeshRenderer>().material;
                longDepthMediaTexture = new Texture2D(320, 288, TextureFormat.Alpha8, false);
                longDepthMediaMaterial.mainTexture = longDepthMediaTexture;
            }

            if (longAbImagePreviewPlane != null)
            {
                longAbImageMediaMaterial = longAbImagePreviewPlane.GetComponent<MeshRenderer>().material;
                longAbImageMediaTexture = new Texture2D(320, 288, TextureFormat.Alpha8, false);
                longAbImageMediaMaterial.mainTexture = longAbImageMediaTexture;
            }
            depthPreviewPlane.SetActive(false);
            shortAbImagePreviewPlane.SetActive(false);
        }
        

        if (LFPreviewPlane != null)
        {
            LFMediaMaterial = LFPreviewPlane.GetComponent<MeshRenderer>().material;
            LFMediaTexture = new Texture2D(640, 480, TextureFormat.Alpha8, false);
            LFMediaMaterial.mainTexture = LFMediaTexture;
        }

        if (RFPreviewPlane != null)
        {
            RFMediaMaterial = RFPreviewPlane.GetComponent<MeshRenderer>().material;
            RFMediaTexture = new Texture2D(640, 480, TextureFormat.Alpha8, false);
            RFMediaMaterial.mainTexture = RFMediaTexture;
        }

        if (pointCloudRendererGo != null)
        {
            pointCloudRenderer = pointCloudRendererGo.GetComponent<PointCloudRenderer>();
        }

        tcpClient = GetComponent<TCPClient>();

#if ENABLE_WINMD_SUPPORT
        researchMode = new HL2ResearchMode();

        // Depth sensor should be initialized in only one mode
        if (depthSensorMode == DepthSensorMode.LongThrow) researchMode.InitializeLongDepthSensor();
        else if (depthSensorMode == DepthSensorMode.ShortThrow) researchMode.InitializeDepthSensor();
        
        researchMode.InitializeSpatialCamerasFront();
        researchMode.SetReferenceCoordinateSystem(unityWorldOrigin);
        researchMode.SetPointCloudDepthOffset(0);

        // Depth sensor should be initialized in only one mode
        if (depthSensorMode == DepthSensorMode.LongThrow) researchMode.StartLongDepthSensorLoop(enablePointCloud);
        else if (depthSensorMode == DepthSensorMode.ShortThrow) researchMode.StartDepthSensorLoop(enablePointCloud);

        researchMode.StartSpatialCamerasFrontLoop();
#endif
    }

    bool startRealtimePreview = true;
    void LateUpdate()
    {
#if ENABLE_WINMD_SUPPORT
        // update depth map texture
        if (depthSensorMode == DepthSensorMode.ShortThrow && startRealtimePreview && 
            depthPreviewPlane != null && researchMode.DepthMapTextureUpdated())
        {
            byte[] frameTexture = researchMode.GetDepthMapTextureBuffer();
            if (frameTexture.Length > 0)
            {
                if (depthFrameData == null)
                {
                    depthFrameData = frameTexture;
                }
                else
                {
                    System.Buffer.BlockCopy(frameTexture, 0, depthFrameData, 0, depthFrameData.Length);
                }

                depthMediaTexture.LoadRawTextureData(depthFrameData);
                depthMediaTexture.Apply();
            }
        }

        // update short-throw AbImage texture
        if (depthSensorMode == DepthSensorMode.ShortThrow && startRealtimePreview && 
            shortAbImagePreviewPlane != null && researchMode.ShortAbImageTextureUpdated())
        {
            byte[] frameTexture = researchMode.GetShortAbImageTextureBuffer();

            
            if(uvMap == null || uvMap.Length == 0) 
            {
                uvMap = researchMode.GetUVPlane();

                text.text = "Depth Offset : " + researchMode.GetDepthOffset().ToString(); 
            }
            
            //using var uv_vals = researchMode.GetUVPlane()
            /*
            //using var matrix = new Mat(512, 512, MatType.CV_8UC1);
            byte[] tmp_data = frameTexture;
            using var imageData = new Mat(tmp_data.Length, 1, MatType.CV_8UC1, tmp_data);
            using var reshapedMat = imageData.Reshape(1, 512);
            using var binaryImage = new Mat();
            //int kernelSize = 7;
            //CvXImgProc.NiblackThreshold(reshapedMat, dst, 255, ThresholdTypes.Binary, kernelSize, -0.2, LocalBinarizationMethods.Niblack);
            
            //Cv2.Threshold(reshapedMat, binaryImage, 15, 255, ThresholdTypes.Binary);
            

            // Define parameters of blob detector
            var circleParams = new SimpleBlobDetector.Params
            {
                MinThreshold = 10,
                MaxThreshold = 230,

                // The area is the number of pixels in the blob.
                FilterByArea = true,
                MinArea = 500,
                MaxArea = 50000,

                // Circularity is a ratio of the area to the perimeter. Polygons with more sides are more circular.
                FilterByCircularity = true,
                MinCircularity = 0.9f,

                // Convexity is the ratio of the area of the blob to the area of its convex hull.
                FilterByConvexity = true,
                MinConvexity = 0.95f,

                // A circle's inertia ratio is 1. A line's is 0. An oval is between 0 and 1.
                FilterByInertia = true,
                MinInertiaRatio = 0.95f
            };
            
            // Create blob detector
            using var circleDetector = SimpleBlobDetector.Create(circleParams);
            
            // Detect keyponts
            var keypoints = circleDetector.Detect(binaryImage);

            //Get world positions of keypoints
            List<List<float>> positions = new List<List<float>>();

            if(keypoints != null && keypoints.Length > 0)
            {
                for(int i = 0; i<keypoints.Length; i++)
                {
                    var x = (int)keypoints[i].Pt.X;
                    var y = (int)keypoints[i].Pt.Y;
                    var position = researchMode.GetWorldPosition(x, y);
                    //positions.Add(position);
                }
                
            }
            
            */
            //int bufferSize = 1 * 512 * 512;
            //byte[] frameTexture = new byte[];
            // TODO: Find right method to convert Mat to byte array
            //dst.GetArray(out byte[] frameTexture);
            
            

            if (frameTexture.Length > 0)
            {
                if (shortAbImageFrameData == null)
                {
                    shortAbImageFrameData = frameTexture;
                }
                else
                {
                    System.Buffer.BlockCopy(frameTexture, 0, shortAbImageFrameData, 0, shortAbImageFrameData.Length);
                }

                shortAbImageMediaTexture.LoadRawTextureData(shortAbImageFrameData);
                shortAbImageMediaTexture.Apply();
            }
        }

        // update long depth map texture
        if (depthSensorMode == DepthSensorMode.LongThrow && startRealtimePreview && 
            longDepthPreviewPlane != null && researchMode.LongDepthMapTextureUpdated())
        {
            byte[] frameTexture = researchMode.GetLongDepthMapTextureBuffer();
            if (frameTexture.Length > 0)
            {
                if (longDepthFrameData == null)
                {
                    longDepthFrameData = frameTexture;
                }
                else
                {
                    System.Buffer.BlockCopy(frameTexture, 0, longDepthFrameData, 0, longDepthFrameData.Length);
                }

                longDepthMediaTexture.LoadRawTextureData(longDepthFrameData);
                longDepthMediaTexture.Apply();
            }
        }

        // update long-throw AbImage texture
        if (depthSensorMode == DepthSensorMode.LongThrow && startRealtimePreview &&
            longAbImagePreviewPlane != null && researchMode.LongAbImageTextureUpdated())
        {
            byte[] frameTexture = researchMode.GetLongAbImageTextureBuffer();
            if (frameTexture.Length > 0)
            {
                if (longAbImageFrameData == null)
                {
                    longAbImageFrameData = frameTexture;
                }
                else
                {
                    System.Buffer.BlockCopy(frameTexture, 0, longAbImageFrameData, 0, longAbImageFrameData.Length);
                }

                longAbImageMediaTexture.LoadRawTextureData(longAbImageFrameData);
                longAbImageMediaTexture.Apply();
            }
        }

        // update LF camera texture
        if (startRealtimePreview && LFPreviewPlane != null && researchMode.LFImageUpdated())
        {
            long ts;
            byte[] frameTexture = researchMode.GetLFCameraBuffer(out ts);
            if (frameTexture.Length > 0)
            {
                if (LFFrameData == null)
                {
                    LFFrameData = frameTexture;
                }
                else
                {
                    System.Buffer.BlockCopy(frameTexture, 0, LFFrameData, 0, LFFrameData.Length);
                }

                LFMediaTexture.LoadRawTextureData(LFFrameData);
                LFMediaTexture.Apply();
            }
        }
        // update RF camera texture
        if (startRealtimePreview && RFPreviewPlane != null && researchMode.RFImageUpdated())
        {
            long ts;
            byte[] frameTexture = researchMode.GetRFCameraBuffer(out ts);
            if (frameTexture.Length > 0)
            {
                if (RFFrameData == null)
                {
                    RFFrameData = frameTexture;
                }
                else
                {
                    System.Buffer.BlockCopy(frameTexture, 0, RFFrameData, 0, RFFrameData.Length);
                }

                RFMediaTexture.LoadRawTextureData(RFFrameData);
                RFMediaTexture.Apply();
            }
        }

        // Update point cloud
        UpdatePointCloud();
#endif
    }

#if ENABLE_WINMD_SUPPORT
    private void UpdatePointCloud()
    {
        if (enablePointCloud && renderPointCloud && pointCloudRendererGo != null)
        {
            if ((depthSensorMode == DepthSensorMode.LongThrow && !researchMode.LongThrowPointCloudUpdated()) ||
                (depthSensorMode == DepthSensorMode.ShortThrow && !researchMode.PointCloudUpdated())) return;

            float[] pointCloud = new float[] { };
            if (depthSensorMode == DepthSensorMode.LongThrow) pointCloud = researchMode.GetLongThrowPointCloudBuffer();
            else if (depthSensorMode == DepthSensorMode.ShortThrow) pointCloud = researchMode.GetPointCloudBuffer();
            
            if (pointCloud.Length > 0)
            {
                int pointCloudLength = pointCloud.Length / 3;
                Vector3[] pointCloudVector3 = new Vector3[pointCloudLength];
                for (int i = 0; i < pointCloudLength; i++)
                {
                    pointCloudVector3[i] = new Vector3(pointCloud[3 * i], pointCloud[3 * i + 1], pointCloud[3 * i + 2]);
                }
                //text.text = "Point Cloud Length: " + pointCloudVector3.Length.ToString();
                pointCloudRenderer.Render(pointCloudVector3, pointColor);
            }
        }
    }
#endif


    #region Button Event Functions
    public void TogglePreviewEvent()
    {
        startRealtimePreview = !startRealtimePreview;
    }

    bool renderPointCloud = true;
    public void TogglePointCloudEvent()
    {
        renderPointCloud = !renderPointCloud;
        if (renderPointCloud)
        {
            pointCloudRendererGo.SetActive(true);
        }
        else
        {
            pointCloudRendererGo.SetActive(false);
        }
    }

    public void StopSensorsEvent()
    {
#if ENABLE_WINMD_SUPPORT
        researchMode.StopAllSensorDevice();
#endif
        startRealtimePreview = false;
    }

    public void SaveAHATSensorDataEvent()
    {
#if ENABLE_WINMD_SUPPORT
        var depthMap = researchMode.GetDepthMapBuffer();
        var AbImage = researchMode.GetShortAbImageBuffer();
#if WINDOWS_UWP
        if (tcpClient != null)
        {
            tcpClient.SendUINT16Async(depthMap, AbImage);
            //text.text = "sending data";
            //tcpClient.SendMapAsync(uvMap);
            //text.text = "data send";        
        }
#endif
#endif
    }

    public void SaveSpatialImageEvent()
    {
#if ENABLE_WINMD_SUPPORT
#if WINDOWS_UWP
        long ts_ft_left, ts_ft_right;
        var LRFImage = researchMode.GetLRFCameraBuffer(out ts_ft_left, out ts_ft_right);
        Windows.Perception.PerceptionTimestamp ts_left = Windows.Perception.PerceptionTimestampHelper.FromHistoricalTargetTime(DateTime.FromFileTime(ts_ft_left));
        Windows.Perception.PerceptionTimestamp ts_right = Windows.Perception.PerceptionTimestampHelper.FromHistoricalTargetTime(DateTime.FromFileTime(ts_ft_right));

        long ts_unix_left = ts_left.TargetTime.ToUnixTimeMilliseconds();
        long ts_unix_right = ts_right.TargetTime.ToUnixTimeMilliseconds();
        long ts_unix_current = GetCurrentTimestampUnix();

        text.text = "Left: " + ts_unix_left.ToString() + "\n" +
            "Right: " + ts_unix_right.ToString() + "\n" +
            "Current: " + ts_unix_current.ToString();

        if (tcpClient != null)
        {
            tcpClient.SendSpatialImageAsync(LRFImage, ts_unix_left, ts_unix_right);
        }
#endif
#endif
    }

    #endregion
    private void OnApplicationFocus(bool focus)
    {
        if (!focus) StopSensorsEvent();
    }

#if WINDOWS_UWP
    private long GetCurrentTimestampUnix()
    {
        // Get the current time, in order to create a PerceptionTimestamp. 
        Windows.Globalization.Calendar c = new Windows.Globalization.Calendar();
        Windows.Perception.PerceptionTimestamp ts = Windows.Perception.PerceptionTimestampHelper.FromHistoricalTargetTime(c.GetDateTime());
        return ts.TargetTime.ToUnixTimeMilliseconds();
        //return ts.SystemRelativeTargetTime.Ticks;
    }
    private Windows.Perception.PerceptionTimestamp GetCurrentTimestamp()
    {
        // Get the current time, in order to create a PerceptionTimestamp. 
        Windows.Globalization.Calendar c = new Windows.Globalization.Calendar();
        return Windows.Perception.PerceptionTimestampHelper.FromHistoricalTargetTime(c.GetDateTime());
    }
#endif
}