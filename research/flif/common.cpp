/*
FLIF - Free Lossless Image Format

Copyright 2010-2016, Jon Sneyers & Pieter Wuille

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "common.hpp"

// These are the names of the transformations done before encoding / after decoding
const char *transforms[] = {"Channel_Compact", "YCoCg", "?? YCbCr ??", "PermutePlanes", "Bounds",  // color space / ranges
                            "Palette_Alpha", "Palette", "Color_Buckets",  // sparse-color transforms
                            "?? DCT ??", "?? DWT ??", // JPEG-style transforms, not implemented (but maybe in future versions?)
                            "Duplicate_Frame", "Frame_Shape", "Frame_Lookback", // animation-related transforms
                            "?? Other ??" };
// Plenty of room for future extensions: transform "Other" can be used to encode identifiers of arbitrary many other transforms

Progress::Progress()
{
    pixels_todo = 0;
    pixels_done = 0;
    progressive_qual_target = 0;
    progressive_qual_shown = -1;
}

// The order in which the planes are encoded.
// Lookback (animations-only, value refers to a previous frame) has to be first, because all other planes are not encoded if lookback != 0
// Alpha has to be next, because for fully transparent A=0 pixels, the other planes are not encoded
// Y (luma) is next (the first channel for still opaque images), because it is perceptually most important
// Co and Cg are in that order because Co is perceptually slightly more important than Cg [citation needed]
const int PLANE_ORDERING[] = {4,3,0,1,2}; // FRA (lookback), A, Y, Co, Cg


// MANIAC property information for non-interlaced images
const int NB_PROPERTIES_scanlines[] = {7,8,9,7,7};
const int NB_PROPERTIES_scanlinesA[] = {8,9,10,7,7};

int nb_properties_scanlines(int p, int nump, int nb_frames, int additional_props) {
    int n = nump > 3 ? NB_PROPERTIES_scanlinesA[p] : NB_PROPERTIES_scanlines[p];
    if (nb_frames > 1) {
      if (is_fr_enabled(additional_props)) n++;
      if (is_pf_miss_enabled(additional_props)) n++;
      if (is_pf_tl_enabled(additional_props)) n++;
    }
    return n;
}

void initPropRanges_scanlines(PropNamesAndRanges &propRanges, const ColorRanges &ranges, int p, int nb_frames, int additional_props) {
    int min = ranges.min(p);
    int max = ranges.max(p);
    int mind = min - max, maxd = max - min;

    if (p < 3) {
      if (p > 0) propRanges.push_back("X0", std::make_pair(ranges.min(0), ranges.max(0)));
      if (p > 1) propRanges.push_back("X1", std::make_pair(ranges.min(1), ranges.max(1)));
      if (ranges.numPlanes()>3) propRanges.push_back("X3", std::make_pair(ranges.min(3), ranges.max(3)));  // pixel on alpha plane
    }
    propRanges.push_back("P", std::make_pair(min,max));   // guess (median of 3)
    propRanges.push_back("Mi", std::make_pair(0,2));      // which predictor was it
    propRanges.push_back("L-TL", std::make_pair(mind,maxd));
    propRanges.push_back("TL-T", std::make_pair(mind,maxd));
    propRanges.push_back("T-TR", std::make_pair(mind,maxd));
    propRanges.push_back("TT-T", std::make_pair(mind,maxd));
    propRanges.push_back("LL-L", std::make_pair(mind,maxd));

    if (nb_frames > 1) {
      if (is_fr_enabled(additional_props))
        propRanges.push_back("Fr", std::make_pair(0,nb_frames-1)); // frame index
      if (is_pf_miss_enabled(additional_props))
        propRanges.push_back("PF Miss", std::make_pair(mind,maxd)); // previous frame prediction miss
      if (is_pf_tl_enabled(additional_props))
        propRanges.push_back("PF L", std::make_pair(mind,maxd)); // left (current frame) - left (previous frame)
    }
}

ColorVal predict_and_calcProps_scanlines(Properties &properties, const ColorRanges *ranges, const Images &images, const int fr, const int p, const uint32_t r, const uint32_t c, ColorVal &min, ColorVal &max, const ColorVal fallback, const bool guessOnly) {
    return predict_and_calcProps_scanlines_plane<GeneralPlane,false>(properties,ranges,images,fr,images.at(fr).getPlane(p),p,r,c,min,max, fallback, guessOnly);
}


// MANIAC property information for interlaced images
const int NB_PROPERTIES[] = {8,10,9,8,8};
const int NB_PROPERTIESA[] = {9,11,10,8,8};

int nb_properties(int p, int nump, int nb_frames, int additional_props) {
    int n = nump > 3 ? NB_PROPERTIESA[p] : NB_PROPERTIES[p];
    if (nb_frames > 1) {
      if (is_fr_enabled(additional_props)) n++;
      if (is_pf_miss_enabled(additional_props)) n++;
      if (is_pf_tl_enabled(additional_props)) n++;
    }
    return n;
}

void initPropRanges(PropNamesAndRanges &propRanges, const ColorRanges &ranges, int p, int nb_frames, int additional_props) {
    int min = ranges.min(p);
    int max = ranges.max(p);
    int mind = min - max, maxd = max - min;
    if (p < 3) {       // alpha channel first
      if (p > 0) propRanges.push_back("X0", std::make_pair(ranges.min(0), ranges.max(0)));
      if (p > 1) propRanges.push_back("X1", std::make_pair(ranges.min(1), ranges.max(1)));
      if (ranges.numPlanes()>3) propRanges.push_back("X3", std::make_pair(ranges.min(3), ranges.max(3)));  // pixel on alpha plane
    }

    //if (p<1 || p>2) 
    propRanges.push_back("Mi", std::make_pair(0,2));       // median predictor: which of the three values is the median?

    if (p==1 || p==2) propRanges.push_back("Luma Miss", std::make_pair(ranges.min(0)-ranges.max(0),ranges.max(0)-ranges.min(0))); // luma prediction miss
    propRanges.push_back("T-B/L-R", std::make_pair(mind,maxd));  // neighbor A - neighbor B   (top-bottom or left-right)
    propRanges.push_back("T/L Miss", std::make_pair(mind,maxd)); // top/left prediction miss (previous pixel)
    propRanges.push_back("L/T Miss", std::make_pair(mind,maxd)); // left/top prediction miss (other direction)
    propRanges.push_back("B/R Miss", std::make_pair(mind,maxd)); // bottom/right prediction miss
    propRanges.push_back("P", std::make_pair(min,max));          // guess

//    propRanges.push_back(std::make_pair(mind,maxd));  // left - topleft
//    propRanges.push_back(std::make_pair(mind,maxd));  // topleft - top

//    if (p == 0 || p > 2)
//      propRanges.push_back(std::make_pair(mind,maxd)); // top - topright
    if (p != 2) {
      propRanges.push_back("TT-T", std::make_pair(mind,maxd)); // toptop - top
      propRanges.push_back("LL-L", std::make_pair(mind,maxd)); // leftleft - left
    }

    if (nb_frames > 1) {
      if (is_fr_enabled(additional_props))
        propRanges.push_back("Fr", std::make_pair(0,nb_frames-1)); // frame index
      if (is_pf_miss_enabled(additional_props))
        propRanges.push_back("PF Miss", std::make_pair(mind,maxd)); // previous frame prediction miss
      if (is_pf_tl_enabled(additional_props))
        propRanges.push_back("PF L", std::make_pair(mind,maxd)); // top/left (current frame) - top/left (previous frame)
    }
}

// Actual prediction. Also sets properties. Property vector should already have the right size before calling this.
// This is a fall-back function which should be replaced by direct calls to the specific predict_and_calcProps_plane function
ColorVal predict_and_calcProps(Properties &properties, const ColorRanges *ranges, const Images &images, const int fr, const int z, const int p, const uint32_t r, const uint32_t c, ColorVal &min, ColorVal &max, const int predictor, const int additional_props) ATTRIBUTE_HOT;
ColorVal predict_and_calcProps(Properties &properties, const ColorRanges *ranges, const Images &images, const int fr, const int z, const int p, const uint32_t r, const uint32_t c, ColorVal &min, ColorVal &max, const int predictor, const int additional_props) {
    const Image &image = images.at(fr);
    image.getPlane(0).prepare_zoomlevel(z);
    image.getPlane(p).prepare_zoomlevel(z);

#ifdef SUPPORT_HDR
    if (image.getDepth() > 8) {
     switch(p) {
      case 0:
        if (z%2==0) return predict_and_calcProps_plane<Plane<ColorVal_intern_16u>,Plane<ColorVal_intern_16u>,true,false,0,ColorRanges>(properties,ranges,images,fr,static_cast<const Plane<ColorVal_intern_16u>&>(image.getPlane(p)),static_cast<const Plane<ColorVal_intern_16u>&>(image.getPlane(0)),z,r,c,min,max,predictor,additional_props);
        else return predict_and_calcProps_plane<Plane<ColorVal_intern_16u>,Plane<ColorVal_intern_16u>,false,false,0,ColorRanges>(properties,ranges,images,fr,static_cast<const Plane<ColorVal_intern_16u>&>(image.getPlane(p)),static_cast<const Plane<ColorVal_intern_16u>&>(image.getPlane(0)),z,r,c,min,max,predictor,additional_props);
      case 1:
        if (z%2==0) return predict_and_calcProps_plane<Plane<ColorVal_intern_32>,Plane<ColorVal_intern_16u>,true,false,1,ColorRanges>(properties,ranges,images,fr,static_cast<const Plane<ColorVal_intern_32>&>(image.getPlane(p)),static_cast<const Plane<ColorVal_intern_16u>&>(image.getPlane(0)),z,r,c,min,max,predictor,additional_props);
        else return predict_and_calcProps_plane<Plane<ColorVal_intern_32>,Plane<ColorVal_intern_16u>,false,false,1,ColorRanges>(properties,ranges,images,fr,static_cast<const Plane<ColorVal_intern_32>&>(image.getPlane(p)),static_cast<const Plane<ColorVal_intern_16u>&>(image.getPlane(0)),z,r,c,min,max,predictor,additional_props);
      case 2:
        if (z%2==0) return predict_and_calcProps_plane<Plane<ColorVal_intern_32>,Plane<ColorVal_intern_16u>,true,false,2,ColorRanges>(properties,ranges,images,fr,static_cast<const Plane<ColorVal_intern_32>&>(image.getPlane(p)),static_cast<const Plane<ColorVal_intern_16u>&>(image.getPlane(0)),z,r,c,min,max,predictor,additional_props);
        else return predict_and_calcProps_plane<Plane<ColorVal_intern_32>,Plane<ColorVal_intern_16u>,false,false,2,ColorRanges>(properties,ranges,images,fr,static_cast<const Plane<ColorVal_intern_32>&>(image.getPlane(p)),static_cast<const Plane<ColorVal_intern_16u>&>(image.getPlane(0)),z,r,c,min,max,predictor,additional_props);
      case 3:
        if (z%2==0) return predict_and_calcProps_plane<Plane<ColorVal_intern_16u>,Plane<ColorVal_intern_16u>,true,false,3,ColorRanges>(properties,ranges,images,fr,static_cast<const Plane<ColorVal_intern_16u>&>(image.getPlane(p)),static_cast<const Plane<ColorVal_intern_16u>&>(image.getPlane(0)),z,r,c,min,max,predictor,additional_props);
        else return predict_and_calcProps_plane<Plane<ColorVal_intern_16u>,Plane<ColorVal_intern_16u>,false,false,3,ColorRanges>(properties,ranges,images,fr,static_cast<const Plane<ColorVal_intern_16u>&>(image.getPlane(p)),static_cast<const Plane<ColorVal_intern_16u>&>(image.getPlane(0)),z,r,c,min,max,predictor,additional_props);
      default:
        assert(p==4);
        if (z%2==0) return predict_and_calcProps_plane<Plane<ColorVal_intern_8>,Plane<ColorVal_intern_16u>,true,false,4,ColorRanges>(properties,ranges,images,fr,static_cast<const Plane<ColorVal_intern_8>&>(image.getPlane(p)),static_cast<const Plane<ColorVal_intern_16u>&>(image.getPlane(0)),z,r,c,min,max,predictor,additional_props);
        else return predict_and_calcProps_plane<Plane<ColorVal_intern_8>,Plane<ColorVal_intern_16u>,false,false,4,ColorRanges>(properties,ranges,images,fr,static_cast<const Plane<ColorVal_intern_8>&>(image.getPlane(p)),static_cast<const Plane<ColorVal_intern_16u>&>(image.getPlane(0)),z,r,c,min,max,predictor,additional_props);
     }
    } else
#endif
     switch(p) {
      case 0:
        if (z%2==0) return predict_and_calcProps_plane<Plane<ColorVal_intern_8>,Plane<ColorVal_intern_8>,true,false,0,ColorRanges>(properties,ranges,images,fr,static_cast<const Plane<ColorVal_intern_8>&>(image.getPlane(p)),static_cast<const Plane<ColorVal_intern_8>&>(image.getPlane(0)),z,r,c,min,max,predictor,additional_props);
        else return predict_and_calcProps_plane<Plane<ColorVal_intern_8>,Plane<ColorVal_intern_8>,false,false,0,ColorRanges>(properties,ranges,images,fr,static_cast<const Plane<ColorVal_intern_8>&>(image.getPlane(p)),static_cast<const Plane<ColorVal_intern_8>&>(image.getPlane(0)),z,r,c,min,max,predictor,additional_props);
      case 1:
        if (image.getPlane(0).is_constant()) {
          if (z%2==0) return predict_and_calcProps_plane<Plane<ColorVal_intern_8>,ConstantPlane,true,false,1,ColorRanges>(properties,ranges,images,fr,static_cast<const Plane<ColorVal_intern_8>&>(image.getPlane(p)),static_cast<const ConstantPlane&>(image.getPlane(0)),z,r,c,min,max,predictor,additional_props);
          else return predict_and_calcProps_plane<Plane<ColorVal_intern_8>,ConstantPlane,false,false,1,ColorRanges>(properties,ranges,images,fr,static_cast<const Plane<ColorVal_intern_8>&>(image.getPlane(p)),static_cast<const ConstantPlane&>(image.getPlane(0)),z,r,c,min,max,predictor,additional_props);
        } else {
          if (z%2==0) return predict_and_calcProps_plane<Plane<ColorVal_intern_16>,Plane<ColorVal_intern_8>,true,false,1,ColorRanges>(properties,ranges,images,fr,static_cast<const Plane<ColorVal_intern_16>&>(image.getPlane(p)),static_cast<const Plane<ColorVal_intern_8>&>(image.getPlane(0)),z,r,c,min,max,predictor,additional_props);
          else return predict_and_calcProps_plane<Plane<ColorVal_intern_16>,Plane<ColorVal_intern_8>,false,false,1,ColorRanges>(properties,ranges,images,fr,static_cast<const Plane<ColorVal_intern_16>&>(image.getPlane(p)),static_cast<const Plane<ColorVal_intern_8>&>(image.getPlane(0)),z,r,c,min,max,predictor,additional_props);
        }
      case 2:
        if (z%2==0) return predict_and_calcProps_plane<Plane<ColorVal_intern_16>,Plane<ColorVal_intern_8>,true,false,2,ColorRanges>(properties,ranges,images,fr,static_cast<const Plane<ColorVal_intern_16>&>(image.getPlane(p)),static_cast<const Plane<ColorVal_intern_8>&>(image.getPlane(0)),z,r,c,min,max,predictor,additional_props);
        else return predict_and_calcProps_plane<Plane<ColorVal_intern_16>,Plane<ColorVal_intern_8>,false,false,2,ColorRanges>(properties,ranges,images,fr,static_cast<const Plane<ColorVal_intern_16>&>(image.getPlane(p)),static_cast<const Plane<ColorVal_intern_8>&>(image.getPlane(0)),z,r,c,min,max,predictor,additional_props);
      case 3:
        if (z%2==0) return predict_and_calcProps_plane<Plane<ColorVal_intern_8>,Plane<ColorVal_intern_8>,true,false,3,ColorRanges>(properties,ranges,images,fr,static_cast<const Plane<ColorVal_intern_8>&>(image.getPlane(p)),static_cast<const Plane<ColorVal_intern_8>&>(image.getPlane(0)),z,r,c,min,max,predictor,additional_props);
        else return predict_and_calcProps_plane<Plane<ColorVal_intern_8>,Plane<ColorVal_intern_8>,false,false,3,ColorRanges>(properties,ranges,images,fr,static_cast<const Plane<ColorVal_intern_8>&>(image.getPlane(p)),static_cast<const Plane<ColorVal_intern_8>&>(image.getPlane(0)),z,r,c,min,max,predictor,additional_props);
      default:
        assert(p==4);
        if (z%2==0) return predict_and_calcProps_plane<Plane<ColorVal_intern_8>,Plane<ColorVal_intern_8>,true,false,4,ColorRanges>(properties,ranges,images,fr,static_cast<const Plane<ColorVal_intern_8>&>(image.getPlane(p)),static_cast<const Plane<ColorVal_intern_8>&>(image.getPlane(0)),z,r,c,min,max,predictor,additional_props);
        else return predict_and_calcProps_plane<Plane<ColorVal_intern_8>,Plane<ColorVal_intern_8>,false,false,4,ColorRanges>(properties,ranges,images,fr,static_cast<const Plane<ColorVal_intern_8>&>(image.getPlane(p)),static_cast<const Plane<ColorVal_intern_8>&>(image.getPlane(0)),z,r,c,min,max,predictor,additional_props);
     }
}

int plane_zoomlevels(const Image &image, const int beginZL, const int endZL) {
    return image.numPlanes() * (beginZL - endZL + 1);
}

std::pair<int, int> plane_zoomlevel(const Image &image, const int beginZL, const int endZL, int i, const ColorRanges *ranges) {
    assert(i >= 0);
    assert(i < plane_zoomlevels(image, beginZL, endZL));
    // simple order: interleave planes, zoom in
//    int p = i % image.numPlanes();
//    int zl = beginZL - (i / image.numPlanes());

    // more advanced order: give priority to more important plane(s)
    // assumption: plane 0 is luma, plane 1 is chroma, plane 2 is less important chroma, plane 3 is perhaps alpha, plane 4 are frame lookbacks (FRA transform, animation only)
    int max_behind[] = {0, 2, 4, 0, 0};

    // if there is no info in the luma plane, there's no reason to lag chroma behind luma
    // (this also happens for palette images)
    if (ranges->min(0) >= ranges->max(0)) {
        max_behind[1] = 0;
        max_behind[2] = 1;
    }
    int np = image.numPlanes();
    if (np>5) {
      // too many planes, do something simple
      int p = i % image.numPlanes();
      int zl = beginZL - (i / image.numPlanes());
      return std::pair<int, int>(p,zl);
    }
    std::vector<int> czl(np);
    for (int &pzl : czl) pzl = beginZL+1;
    int highest_priority_plane = 0;
    if (np >= 4) highest_priority_plane = 3; // alpha first
    if (np >= 5) highest_priority_plane = 4; // lookbacks first
    int nextp = highest_priority_plane;
    while (i >= 0) {
      czl[nextp]--;
      i--;
      if (i<0) break;
      nextp=highest_priority_plane;
      for (int p=0; p<np; p++) {
        if (czl[p] > czl[highest_priority_plane] + max_behind[p]) {
          nextp = p; //break;
        }
      }
      // ensure that nextp is not at the most detailed zoomlevel yet
      while (czl[nextp] <= endZL) nextp = (nextp+1)%np;
    }
    int p = nextp;
    int zl = czl[p];

    return std::pair<int, int>(p,zl);
}

void print_tree(const int p, const Tree &tree, const std::vector<const char*> &propNames) {
    FILE *f = stderr;
    fprintf(f, "digraph P%d {\n", p);

    for (uint32_t i = 0; i < tree.size(); i++) {
        const auto &node = tree[i];
        if (node.property == -1) {
            fprintf(f, "N%04u [label=Leaf];\n", i);
        } else {
            const char *propName = propNames.at(node.property);
            fprintf(f, "N%04u [label=\"%s > %d\\nCount: %d\"];\n", i, propName, node.splitval, node.count);
            fprintf(f, "N%04u -> N%04u [label=\">\"];\nN%04u -> N%04u [label=\"<=\"];\n", i, node.childID, i, node.childID + 1);
        }
    }

    fprintf(f, "}\n");
}
