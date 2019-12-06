/**
* This file is a modified version of ORB-SLAM2.<https://github.com/raulmur/ORB_SLAM2>
*
* This file is part of DynaSLAM.
* Copyright (C) 2018 Berta Bescos <bbescos at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/bertabescos/DynaSLAM>.
*
*/

#include "Frame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include <thread>

namespace ORB_SLAM2
{

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

const int LABEL_BACKGROUND=0;

std::vector<float> labelWeight={
        /*0 */    0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
        /*10*/    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
        /*20*/    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
        /*30*/    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
        /*40*/    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
        /*50*/    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.4,0.0,0.0,
        /*60*/    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
        /*70*/    0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,
        /*70*/    0.0   
            };


Frame::Frame()
{}

//Copy Constructor
Frame::Frame(const Frame &frame)
    :mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn),  mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
     mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels), mImRGB(frame.mImRGB),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor), mImGray(frame.mImGray),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),mImMask(frame.mImMask),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2),mIsKeyFrame(frame.mIsKeyFrame),mImDepth(frame.mImDepth)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j]=frame.mGrid[i][j];

    if(!frame.mTcw.empty())
        SetPose(frame.mTcw);
}


Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const cv::Mat &maskLeft, const cv::Mat &maskRight,const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mpReferenceKF(static_cast<KeyFrame*>(NULL))
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft);
    thread threadRight(&Frame::ExtractORB,this,1,imRight);
    threadLeft.join();
    threadRight.join();

    // Delete those ORB points that fall in Mask borders (Included by Berta)
    cv::Mat MaskLeft_dil = maskLeft.clone();
    cv::Mat MaskRight_dil = maskRight.clone();
    int dilation_size = 15;
    cv::Mat kernel = getStructuringElement(cv::MORPH_ELLIPSE,
                                        cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                        cv::Point( dilation_size, dilation_size ) );
    cv::erode(maskLeft, MaskLeft_dil, kernel);
    cv::erode(maskRight, MaskRight_dil, kernel);

    if(mvKeys.empty())
        return;

    std::vector<cv::KeyPoint> _mvKeys;
    cv::Mat _mDescriptors;

    for (size_t i(0); i < mvKeys.size(); ++i)
    {
        int val = (int)MaskLeft_dil.at<uchar>(mvKeys[i].pt.y,mvKeys[i].pt.x);
        if (val == 1)
        {
            _mvKeys.push_back(mvKeys[i]);
            _mDescriptors.push_back(mDescriptors.row(i));
        }
    }

    mvKeys = _mvKeys;
    mDescriptors = _mDescriptors;

    std::vector<cv::KeyPoint> _mvKeysRight;
    cv::Mat _mDescriptorsRight;

    for (size_t i(0); i < mvKeysRight.size(); ++i)
    {
        int val = (int)MaskRight_dil.at<uchar>(mvKeysRight[i].pt.y,mvKeysRight[i].pt.x);
        if (val == 1)
        {
            _mvKeysRight.push_back(mvKeysRight[i]);
            _mDescriptorsRight.push_back(mDescriptorsRight.row(i));
        }
    }

    mvKeysRight = _mvKeysRight;
    mDescriptorsRight = _mDescriptorsRight;


    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoMatches();

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));    
    mvbOutlier = vector<bool>(N,false);


    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const cv::Mat &imMask, const double &timeStamp,  ORBextractor* extractor, ORBVocabulary* voc,
             cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mImMask(imMask), mpORBvocabulary(voc), mpORBextractorLeft(extractor), mpORBextractorRight(static_cast<ORBextractor*>(NULL)), mImGray(imGray),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth), mIsKeyFrame(false), mImDepth(imDepth)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();    
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(0,imGray);

    // Delete those ORB points that fall in Mask borders (Included by Berta)
    cv::Mat Mask_dil = imMask.clone();
    int dilation_size = 15;
    cv::Mat kernel = getStructuringElement(cv::MORPH_ELLIPSE,
                                        cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                        cv::Point( dilation_size, dilation_size ) );
    cv::erode(imMask, Mask_dil, kernel);

    if(mvKeys.empty())
        return;

    std::vector<cv::KeyPoint> _mvKeys;
    cv::Mat _mDescriptors;

    for (size_t i(0); i < mvKeys.size(); ++i)
    {
        int val = (int)Mask_dil.at<uchar>(mvKeys[i].pt.y,mvKeys[i].pt.x);
        if (val == 1)
        {
            _mvKeys.push_back(mvKeys[i]);
            _mDescriptors.push_back(mDescriptors.row(i));
        }
    }

    mvKeys = _mvKeys;
    mDescriptors = _mDescriptors;

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoFromRGBD(imDepth);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const cv::Mat &imMask, const cv::Mat &imRGB,
             const double &timeStamp,  ORBextractor* extractor, ORBVocabulary* voc,
             cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mImMask(imMask), mpORBvocabulary(voc), mpORBextractorLeft(extractor), mpORBextractorRight(static_cast<ORBextractor*>(NULL)), mImGray(imGray),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth), mIsKeyFrame(false), mImDepth(imDepth), mImRGB(imRGB)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(0,imGray);

    // Delete those ORB points that fall in Mask borders (Included by Berta)
    cv::Mat Mask_dil = imMask.clone();
    int dilation_size = 15;
    cv::Mat kernel = getStructuringElement(cv::MORPH_ELLIPSE,
                                        cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                        cv::Point( dilation_size, dilation_size ) );
    cv::erode(imMask, Mask_dil, kernel);

    if(mvKeys.empty())
        return;

    std::vector<cv::KeyPoint> _mvKeys;
    cv::Mat _mDescriptors;

    for (size_t i(0); i < mvKeys.size(); ++i)
    {
        int val = (int)Mask_dil.at<uchar>(mvKeys[i].pt.y,mvKeys[i].pt.x);
        if (val == 1)
        {
            _mvKeys.push_back(mvKeys[i]);
            _mDescriptors.push_back(mDescriptors.row(i));
        }
    }

    mvKeys = _mvKeys;
    mDescriptors = _mDescriptors;

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoFromRGBD(imDepth);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

// modified by kylin
Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const cv::Mat &imMask, const cv::Mat &imMaskColor, const std::vector<cv::Rect> &imROIs, const std::vector<int> &imClassIds, 
            const cv::Mat &imRGB, const double &timeStamp,  ORBextractor* extractor, ORBVocabulary* voc,
             cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mImMask(imMask),mImMaskColor(imMaskColor), mImROIs(imROIs), mpORBvocabulary(voc), mpORBextractorLeft(extractor), mpORBextractorRight(static_cast<ORBextractor*>(NULL)), mImGray(imGray),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth), mIsKeyFrame(false), mImDepth(imDepth), mImRGB(imRGB)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    flagMovePolar=false;
    flagMoveSemantic=false;
    // ORB extraction
    // ExtractORB(0,imGray);
    ExtractORBKeyPoints(0,imGray);

    // Delete those ORB points that fall in Mask borders (Included by Berta)
    cv::Mat Mask_dil = imMask.clone();
    int dilation_size = 15;
    cv::Mat kernel = getStructuringElement(cv::MORPH_ELLIPSE,
                                        cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                        cv::Point( dilation_size, dilation_size ) );
    cv::erode(imMask, Mask_dil, kernel);

    if(mvKeysTemp.empty())
        return;
    InitSegData(imROIs,imClassIds);
    // dynaslam的滤出动态点方法
    // std::vector<cv::KeyPoint> _mvKeys;
    // cv::Mat _mDescriptors;
    // // 在这里就把动态范围的点滤出了
    // for (size_t i(0); i < mvKeys.size(); ++i)
    // {
    //    //    cout<<"test finishi"<<endl;
    //     int val = (int)Mask_dil.at<uchar>(mvKeys[i].pt.y,mvKeys[i].pt.x);
    //     //   cout<<"test finishi2"<<endl;
    //     if (val == 1)
    //     {
    //         _mvKeys.push_back(mvKeys[i]);
    //         _mDescriptors.push_back(mDescriptors.row(i));
    //     }
    // }
    // mvKeys = _mvKeys;
    // mDescriptors = _mDescriptors;

    // 运动物体检测
    cv::Mat imGrayTemp=imGray.clone();
    flagMoveSemantic=MovingCheckBySemantic();
    if(flagMoveSemantic&&mImGrayPre.data)
    {
        flagMovePolar=MovingCheckByPolar();
    }
    else
    {
        flagMovePolar=false;
    }
    std::swap(mImGrayPre,imGrayTemp);

    if(flagMovePolar&&flagMoveSemantic)
        mpORBextractorLeft->FilterMovingPoint(mImSegData,imMask,mvKeysTemp);

    ExtractORBDesp(0,imGray);
    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoFromRGBD(imDepth);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

Frame::Frame(const cv::Mat &imGray, const cv::Mat &mask, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(0,imGray);

    // Delete those ORB points that fall in mask borders
    cv::Mat Mask_dil = mask.clone();
    int dilation_size = 15;
    cv::Mat kernel = getStructuringElement(cv::MORPH_ELLIPSE,
                                        cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                        cv::Point( dilation_size, dilation_size ) );
    cv::erode(mask, Mask_dil, kernel);

    if(mvKeys.empty())
        return;

    std::vector<cv::KeyPoint> _mvKeys;
    cv::Mat _mDescriptors;

    for (size_t i(0); i < mvKeys.size(); ++i)
    {
        int val = (int)Mask_dil.at<uchar>(mvKeys[i].pt.y,mvKeys[i].pt.x);
        if (val == 1)
        {
            _mvKeys.push_back(mvKeys[i]);
            _mDescriptors.push_back(mDescriptors.row(i));
        }
    }

    mvKeys = _mvKeys;
    mDescriptors = _mDescriptors;

    if(mvKeys.empty())
        return;

    N = mvKeys.size();

    UndistortKeyPoints();

    // Set no stereo information
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

// Frame::Frame(const cv::Mat &imGray, const cv::Mat &mask, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, const float &imMaskColor)
//     :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
//      mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth), mImMaskColor(imMaskColor)
// {
//     // Frame ID
//     mnId=nNextId++;

//     // Scale Level Info
//     mnScaleLevels = mpORBextractorLeft->GetLevels();
//     mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
//     mfLogScaleFactor = log(mfScaleFactor);
//     mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
//     mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
//     mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
//     mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

//     // ORB extraction
//     ExtractORB(0,imGray);

//     // Delete those ORB points that fall in mask borders
//     cv::Mat Mask_dil = mask.clone();
//     int dilation_size = 15;
//     cv::Mat kernel = getStructuringElement(cv::MORPH_ELLIPSE,
//                                         cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
//                                         cv::Point( dilation_size, dilation_size ) );
//     cv::erode(mask, Mask_dil, kernel);

//     if(mvKeys.empty())
//         return;

//     std::vector<cv::KeyPoint> _mvKeys;
//     cv::Mat _mDescriptors;

//     for (size_t i(0); i < mvKeys.size(); ++i)
//     {
//         int val = (int)Mask_dil.at<uchar>(mvKeys[i].pt.y,mvKeys[i].pt.x);
//         if (val == 1)
//         {
//             _mvKeys.push_back(mvKeys[i]);
//             _mDescriptors.push_back(mDescriptors.row(i));
//         }
//     }

//     mvKeys = _mvKeys;
//     mDescriptors = _mDescriptors;

//     if(mvKeys.empty())
//         return;

//     N = mvKeys.size();

//     UndistortKeyPoints();

//     // Set no stereo information
//     mvuRight = vector<float>(N,-1);
//     mvDepth = vector<float>(N,-1);

//     mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
//     mvbOutlier = vector<bool>(N,false);

//     // This is done only for the first Frame (or after a change in the calibration)
//     if(mbInitialComputations)
//     {
//         ComputeImageBounds(imGray);

//         mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
//         mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

//         fx = K.at<float>(0,0);
//         fy = K.at<float>(1,1);
//         cx = K.at<float>(0,2);
//         cy = K.at<float>(1,2);
//         invfx = 1.0f/fx;
//         invfy = 1.0f/fy;

//         mbInitialComputations=false;
//     }

//     mb = mbf/fx;

//     AssignFeaturesToGrid();
// }

void Frame::InitSegData(std::vector<cv::Rect> ROIs,std::vector<int> ClassIds)
{
    //labelWeight[1]=1.0;
    mImSegData.clear();
    for(int i=0;i<ROIs.size();i++)
    {
        int id=ClassIds[i];
        if(labelWeight[id]==0)
            continue;
        else 
        {
            SegData segData;
            segData.mImROI=ROIs[i];  
            segData.classId=id;
            //segData.weight=labelWeight[segData.classId];
            segData.weight=labelWeight[id];
            segData.KeyPointNum=0;
            segData.IsMove=false;
            mImSegData.push_back(segData);    
        }
    }
    
}

bool Frame::MovingCheckByPolar()
{
    // Clear the previous data
	F_prepoint.clear();
	F_nextpoint.clear();
	F2_prepoint.clear();
	F2_nextpoint.clear();
	T_M.clear();

	// Detect dynamic target and ultimately optput the T matrix
	
    cv::goodFeaturesToTrack(mImGrayPre, prepoint, 1000, 0.01, 8, cv::Mat(), 3, true, 0.04);
    cv::cornerSubPix(mImGrayPre, prepoint, cv::Size(10, 10), cv::Size(-1, -1), cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));
	cv::calcOpticalFlowPyrLK(mImGrayPre, mImGray, prepoint, nextpoint, state, err, cv::Size(22, 22), 5, cv::TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.01));

	for (int i = 0; i < state.size(); i++)
    {
        if(state[i] != 0)
        {
            int dx[10] = { -1, 0, 1, -1, 0, 1, -1, 0, 1 };
            int dy[10] = { -1, -1, -1, 0, 0, 0, 1, 1, 1 };
            int x1 = prepoint[i].x, y1 = prepoint[i].y;
            int x2 = nextpoint[i].x, y2 = nextpoint[i].y;
            if ((x1 < limit_edge_corner || x1 >= mImGray.cols - limit_edge_corner || x2 < limit_edge_corner || x2 >= mImGray.cols - limit_edge_corner
            || y1 < limit_edge_corner || y1 >= mImGray.rows - limit_edge_corner || y2 < limit_edge_corner || y2 >= mImGray.rows - limit_edge_corner))
            {
                state[i] = 0;
                continue;
            }
            double sum_check = 0;
            for (int j = 0; j < 9; j++)
                sum_check += abs(mImGrayPre.at<uchar>(y1 + dy[j], x1 + dx[j]) - mImGray.at<uchar>(y2 + dy[j], x2 + dx[j]));
            if (sum_check > limit_of_check) state[i] = 0;
            if (state[i])
            {
                F_prepoint.push_back(prepoint[i]);
                F_nextpoint.push_back(nextpoint[i]);
            }
        }
    }
    // F-Matrix
    cv::Mat mask = cv::Mat(cv::Size(1, 300), CV_8UC1);
    cv::Mat F = cv::findFundamentalMat(F_prepoint, F_nextpoint, mask, cv::FM_RANSAC, 0.1, 0.99);
    for (int i = 0; i < mask.rows; i++)
    {
        if (mask.at<uchar>(i, 0) == 0);
        else
        {
            // Circle(pre_frame, F_prepoint[i], 6, Scalar(255, 255, 0), 3);
            double A = F.at<double>(0, 0)*F_prepoint[i].x + F.at<double>(0, 1)*F_prepoint[i].y + F.at<double>(0, 2);
            double B = F.at<double>(1, 0)*F_prepoint[i].x + F.at<double>(1, 1)*F_prepoint[i].y + F.at<double>(1, 2);
            double C = F.at<double>(2, 0)*F_prepoint[i].x + F.at<double>(2, 1)*F_prepoint[i].y + F.at<double>(2, 2);
            double dd = fabs(A*F_nextpoint[i].x + B*F_nextpoint[i].y + C) / sqrt(A*A + B*B); //Epipolar constraints
            if (dd <= 0.1)
            {
                F2_prepoint.push_back(F_prepoint[i]);
                F2_nextpoint.push_back(F_nextpoint[i]);
            }
        }
    }
    F_prepoint = F2_prepoint;
    F_nextpoint = F2_nextpoint;

    for (int i = 0; i < prepoint.size(); i++)
    {
        if (state[i] != 0)
        {
            double A = F.at<double>(0, 0)*prepoint[i].x + F.at<double>(0, 1)*prepoint[i].y + F.at<double>(0, 2);
            double B = F.at<double>(1, 0)*prepoint[i].x + F.at<double>(1, 1)*prepoint[i].y + F.at<double>(1, 2);
            double C = F.at<double>(2, 0)*prepoint[i].x + F.at<double>(2, 1)*prepoint[i].y + F.at<double>(2, 2);
            double dd = fabs(A*nextpoint[i].x + B*nextpoint[i].y + C) / sqrt(A*A + B*B);

            // Judge outliers
            if (dd <= limit_dis_epi) continue;
            T_M.push_back(nextpoint[i]);
        }
    }

    // add by kylin
    bool hasOutlier=false;
    for (int i = 0; i < prepoint.size(); i++)
    {
        if (state[i] == 0)
            continue;
        for(auto &segData : mImSegData)
        {
            auto roi=segData.mImROI;
            int x1=roi.x;
            int x2=roi.x+roi.width;
            int y1=roi.y;
            int y2=roi.y+roi.height;
            int x=nextpoint[i].x;
            int y=nextpoint[i].y;
            float weight=segData.weight;
            if(x>x1&&x<x2&&y>y1&&y<y2)
            {
                int val=(int)mImMask.ptr<uchar>(x)[y];
                if(val!=segData.classId)
                    continue;
                double A = F.at<double>(0, 0)*prepoint[i].x + F.at<double>(0, 1)*prepoint[i].y + F.at<double>(0, 2);
                double B = F.at<double>(1, 0)*prepoint[i].x + F.at<double>(1, 1)*prepoint[i].y + F.at<double>(1, 2);
                double C = F.at<double>(2, 0)*prepoint[i].x + F.at<double>(2, 1)*prepoint[i].y + F.at<double>(2, 2);
                double dd = fabs(A*nextpoint[i].x + B*nextpoint[i].y + C) / sqrt(A*A + B*B);
                dd*=segData.weight;
                if (dd > limit_dis_epi)
                {
                    segData.T_M.push_back(nextpoint[i]);
                    hasOutlier=true;
                }
                segData.KeyPointNum++;
                break;
            }
        }
    }
    if(!hasOutlier)
        return false;
    else
    {
        return true;
    }

    // if(T_M.empty())
    //     return false;
    // else
    // {
    //     return true;
    // }
    
}

bool Frame::MovingCheckBySemantic()
{
    if(mImSegData.empty())
        return false;
    else
    {
        return true;
    }
    
    // for(auto &segData : mImSegData)
    // {
    //     auto roi=segData.mImROI;
    //     int x1=roi.x;
    //     int x2=roi.x+roi.width;
    //     int y1=roi.y;
    //     int y2=roi.y+roi.height;
    //     std::map<int,int> labelCount;
    //     std::map<int,int>::iterator labelCountIter;
    //     for(int i=x1;i<x2;i++)
    //     {
    //         for(int j=y1;j<y2;j++)
    //         {
    //             int val=(int)mImMask.ptr<uchar>(m)[n];
    //             if(val!=LABEL_BACKGROUND)
    //             {
    //                 if(labelCount.find(val)==labelCount.end())
    //                     labelCount[val]=1;
    //                 else
    //                     labelCount[val]++;
    //             }
    //         }
    //     }
    //     int maxLabel=-1;
    //     int maxLabelCount=-1;
    //     for(labelCountIter=labelCount.begin();labelCountIter!=labelCount.end();labelCountIter++)
    //     {
    //         if(*labelCountIter->second>maxLabelCount)
    //         {
    //             maxLabel=*labelCountIter->first;
    //             maxLabelCount=*labelCountIter->second;
    //         }
    //     }
    //     segData.weight=labelWeight[maxLabel];
    //     segData.label=maxLabel
    // }
}

void Frame::AssignFeaturesToGrid()
{
    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);

    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = mvKeysUn[i];

        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

void Frame::ExtractORB(int flag, const cv::Mat &im)
{
    if(flag==0)
        (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors);
    else
        (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight);
}

void Frame::ExtractORBKeyPoints(int flag,const cv::Mat &imgray)
{
    if(flag==0)
    {
        (*mpORBextractorLeft)( imgray,cv::Mat(),mvKeysTemp);
    }
    else
        (*mpORBextractorRight)(imgray,cv::Mat(),mvKeysTemp);
}

void Frame::ExtractORBDesp(int flag,const cv::Mat &imgray)
{
    if(flag==0)
        (*mpORBextractorLeft).ProcessDesp(imgray,cv::Mat(),mvKeysTemp,mvKeys,mDescriptors);
    else
        (*mpORBextractorLeft).ProcessDesp(imgray,cv::Mat(),mvKeysTemp,mvKeysRight,mDescriptorsRight);
    
}

void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}

void Frame::UpdatePoseMatrices()
{ 
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0,3).col(3);
    mOw = -mRcw.t()*mtcw;
}

bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    pMP->mbTrackInView = false;

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos(); 

    // 3D in camera coordinates
    const cv::Mat Pc = mRcw*P+mtcw;
    const float &PcX = Pc.at<float>(0);
    const float &PcY= Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    const float u=fx*PcX*invz+cx;
    const float v=fy*PcY*invz+cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const cv::Mat PO = P-mOw;
    const float dist = cv::norm(PO);

    if(dist<minDistance || dist>maxDistance)
        return false;

   // Check viewing angle
    cv::Mat Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn)/dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    // Data used by the tracking
    pMP->mbTrackInView = true;
    pMP->mTrackProjX = u;
    pMP->mTrackProjXR = u - mbf*invz;
    pMP->mTrackProjY = v;
    pMP->mnTrackScaleLevel= nPredictedLevel;
    pMP->mTrackViewCos = viewCos;

    return true;
}

vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}


void Frame::ComputeBoW()
{
    if(mBowVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

void Frame::UndistortKeyPoints()
{
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }
}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        // Undistort corners
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));

    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

void Frame::ComputeStereoMatches()
{
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();

    for(int iR=0; iR<Nr; iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Set limits for search
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf/minZ;

    // For each left keypoint search a match in the right image
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;

        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            const float &uR = kpR.pt.x;

            if(uR>=minU && uR<=maxU)
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
            const int w = 5;
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
            IL.convertTo(IL,CV_32F);
            IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1);

            const float iniu = scaleduR0+L-w;
            const float endu = scaleduR0+L+w+1;
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)
            {
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                IR.convertTo(IR,CV_32F);
                IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);

                float dist = cv::norm(IL,IR,cv::NORM_L1);
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }

                vDists[L+incR] = dist;
            }

            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            float disparity = (uL-bestuR);

            if(disparity>=minD && disparity<maxD)
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                mvDepth[iL]=mbf/disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }

    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}


void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v,u);

        if(d>0)
        {
            //cout << "Depth: " << d << " m" << endl;
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;
        }
    }
}

cv::Mat Frame::UnprojectStereo(const int &i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        return mRwc*x3Dc+mOw;
    }
    else
        return cv::Mat();
}

} //namespace ORB_SLAM
