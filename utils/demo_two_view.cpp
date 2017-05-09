/**
 * Date:  2016
 * Author: Rafael Muñoz Salinas
 * Description: demo application of DBoW3
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <vector>

// DBoW3
#include "DBoW3.h"

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#ifdef USE_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif
#include "DescManip.h"

using namespace DBoW3;
using namespace std;


//command line parser
class CmdLineParser{int argc; char **argv; public: CmdLineParser(int _argc,char **_argv):argc(_argc),argv(_argv){}  bool operator[] ( string param ) {int idx=-1;  for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i;    return ( idx!=-1 ) ;    } string operator()(string param,string defvalue="-1"){int idx=-1;    for ( int i=0; i<argc && idx==-1; i++ ) if ( string ( argv[i] ) ==param ) idx=i; if ( idx==-1 ) return defvalue;   else  return ( argv[  idx+1] ); }};


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

// extended surf gives 128-dimensional vectors
const bool EXTENDED_SURF = false;
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void wait()
{
    cout << endl << "Press enter to continue" << endl;
    getchar();
}


vector<string> readImagePaths(int argc,char **argv,int start){
    vector<string> paths;
    for(int i=start;i<argc;i++)    paths.push_back(argv[i]);
        return paths;
}

vector< cv::Mat  >  loadFeatures( std::vector<string> path_to_images,string descriptor, std::vector<std::vector<cv::KeyPoint>>& keypoint, std::vector<cv::Mat> imgs, int nfeatures = 500) throw (std::exception){
    //select detector
    cv::Ptr<cv::Feature2D> fdetector;
    if (descriptor=="orb")        fdetector=cv::ORB::create(nfeatures);
    else if (descriptor=="brisk") fdetector=cv::BRISK::create();
#ifdef OPENCV_VERSION_3
    else if (descriptor=="akaze") fdetector=cv::AKAZE::create();
#endif
#ifdef USE_CONTRIB
    else if(descriptor=="surf" )  fdetector=cv::xfeatures2d::SURF::create(400, 4, 2, EXTENDED_SURF);
    else if(descriptor=="sift") fdetector=cv::xfeatures2d::SIFT::create(400,8,0.04,10,1.6);
#endif

    else throw std::runtime_error("Invalid descriptor");
    assert(!descriptor.empty());
    vector<cv::Mat>    features;


    cout << "Extracting   features..." << endl;
    for(size_t i = 0; i < path_to_images.size(); ++i)
    {
        vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        cout<<"reading image: "<<path_to_images[i]<<endl;
        cv::Mat image = cv::imread(path_to_images[i], 0);
//        cv::resize(image, image, cv::Size(320,240), 0.5,0.5);
//        cv::imshow(string("image") + to_string(i),image);
//        cv::waitKey(0);
        if(image.empty())throw std::runtime_error("Could not open image"+path_to_images[i]);
        imgs.push_back(image);
        cout<<"extracting features"<<endl;
        fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
        features.push_back(descriptors);
        keypoint.push_back(keypoints);
        cout<<"done detecting features"<<endl;
    }
    return features;
}

// ----------------------------------------------------------------------------

void testVocCreation(const vector<cv::Mat> &features)
{
//    // branching factor and depth levels
//    const int k = 9;
//    const int L = 3;
//    const WeightingType weight = TF_IDF;
//    const ScoringType score = L1_NORM;

//    DBoW3::Vocabulary voc(k, L, weight, score);

//    cout << "Creating a small " << k << "^" << L << " vocabulary..." << endl;
//    voc.create(features);
//    cout << "... done!" << endl;

//    cout << "Vocabulary information: " << endl
//         << voc << endl << endl;

//    // lets do something with this vocabulary
//    cout << "Matching images against themselves (0 low, 1 high): " << endl;
//    BowVector v1, v2;
//    for(size_t i = 0; i < features.size(); i++)
//    {
//        voc.transform(features[i], v1);
//        for(size_t j = 0; j < features.size(); j++)
//        {
//            voc.transform(features[j], v2);

//            double score = voc.score(v1, v2);
//            cout << "Image " << i << " vs Image " << j << ": " << score << endl;
//        }
//    }

//    // save the vocabulary to disk
//    cout << endl << "Saving vocabulary..." << endl;
//    voc.save("small_voc.yml.gz");
//    cout << "Done" << endl;
}

////// ----------------------------------------------------------------------------

void testDatabase(const  vector<cv::Mat > &features)
{
    cout << "Creating a small database..." << endl;

    // load the vocabulary from disk
    Vocabulary voc("/home/xushen/Code/DBow3/orbvoc.dbow3");

    Database db(voc, false, 0); // false = do not use direct index
    // (so ignore the last param)
    // The direct index is useful if we want to retrieve the features that
    // belong to some vocabulary node.
    // db creates a copy of the vocabulary, we may get rid of "voc" now

    // add images to the database
    for(size_t i = 0; i < features.size(); i++)
        db.add(features[i]);

    cout << "... done!" << endl;

    cout << "Database information: " << endl << db << endl;

    // and query the database
    cout << "Querying the database: " << endl;

    QueryResults ret;
    for(size_t i = 0; i < features.size(); i++)
    {
        db.query(features[i], ret, 6);

        // ret[0] is always the same image in this case, because we added it to the
        // database. ret[1] is the second best match.

        cout << "Searching for Image " << i << ". " << ret << endl;
    }

    cout << endl;

    // we can save the database. The created file includes the vocabulary
    // and the entries added
//    cout << "Saving database..." << endl;
//    db.save("small_db.yml.gz");
//    cout << "... done!" << endl;

//    // once saved, we can load it again
//    cout << "Retrieving database once again..." << endl;
//    Database db2("small_db.yml.gz");
//    cout << "... done! This is: " << endl << db2 << endl;
}


// ----------------------------------------------------------------------------

int main(int argc,char **argv)
{

    try{
        CmdLineParser cml(argc,argv);
        if (cml["-h"] || argc<=2){
            cerr<<"Usage:  descriptor_name     image0 image1 ... \n\t descriptors:brisk,surf,orb ,akaze(only if using opencv 3)"<<endl;
             return -1;
        }

        string descriptor=argv[1];

        vector<string> images=readImagePaths(argc,argv,2);
        vector<string> image1, image2;
        image1.push_back(images[0]);
        image1.push_back(images[1]);
        image1.push_back(images[2]);
        image1.push_back(images[3]);
        image1.push_back(images[4]);
        image2.push_back(images[5]);
        image2.push_back(images[6]);

        std::vector<std::vector<cv::KeyPoint>> keypoints;
        std::vector< cv::Mat   > imgs;
        std::vector< cv::Mat   >   features1= loadFeatures(image1,"orb",keypoints, imgs);
        std::vector< cv::Mat   >   features2= loadFeatures(image2,"orb",keypoints, imgs, 250);
//        cv::BFMatcher fb;
//        std::vector<cv::DMatch> matches;
//        fb.match(features2[5], features2[6], matches);
//        cout << "match end" << endl;
//        cv::Mat img_match;
//        cv::drawMatches(imgs[5], keypoints[5], imgs[6], keypoints[6], matches, img_match);
//        cv::imshow("test",img_match);
        cv::Mat combf;
        combf.push_back(features2[0]);
        combf.push_back(features2[1]);
        features1.push_back(combf);

        testDatabase(features1);

    }catch(std::exception &ex){
        cerr<<ex.what()<<endl;
    }

    return 0;
}
