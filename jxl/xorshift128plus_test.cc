// Copyright (c) the JPEG XL Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "jxl/xorshift128plus_test.cc"

#include <stdint.h>

#include <algorithm>
#include <vector>

#define HWY_USE_GTEST
#include <hwy/tests/test_util.h>

#include "jxl/base/data_parallel.h"
#include "jxl/base/thread_pool_internal.h"

struct XorshiftTest {
  HWY_DECLARE(void, ())
};
TEST(RationalPolynomialTest, Run) { hwy::RunTests<XorshiftTest>(); }

#endif  // HWY_TARGET_INCLUDE
#include <hwy/tests/test_target_util.h>

namespace jxl {
namespace HWY_NAMESPACE {
namespace {

#include "jxl/xorshift128plus-inl.h"

// Define to nonzero in order to print the (new) golden outputs.
#define PRINT_RESULTS 0

const size_t kVectors = 64;

#if PRINT_RESULTS

template <int kNumLanes>
void Print(const uint64_t (&result)[kNumLanes]) {
  printf("{ ");
  for (int i = 0; i < kNumLanes; ++i) {
    if (i != 0) {
      printf(", ");
    }
    printf("0x%016lXull", result[i]);
  }
  printf("},\n");
}

#else  // PRINT_RESULTS

const uint64_t kExpected[kVectors][Xorshift128Plus::N] = {
    {0x5AC3FDEEA477CBB1ull, 0x643D33C30B5DA2A2ull, 0x10854F048EE0AE99ull,
     0xBA02AE0E62FD6896ull, 0x0643785550DCF532ull, 0x2193ABC1CDDE7E32ull,
     0x8B7C2E7E2A5C0DD0ull, 0xC8177B7BD334D836ull},
    {0x03C3CF0B1ABD3A26ull, 0x276AA22D168B112Dull, 0x18FE330E8E819E36ull,
     0x30959D4F0F05AC53ull, 0x3ADD7E7B3F0783DDull, 0x457E2837FED9B41Aull,
     0xBD5DC597CDBE96B8ull, 0x3D2A8E28934FBD01ull},
    {0xA4D583B0077AC6A2ull, 0x903DD9ADE805E404ull, 0x38FE5E8246EB4CCDull,
     0xCB0FF19182CA899Bull, 0x310B955EF70BF26Aull, 0x2E776FA436C6649Full,
     0x77A6EF9FF4C6B937ull, 0xAC6822F5BD7A353Aull},
    {0x3BD886C30A9C5EB6ull, 0x4484046570B40256ull, 0x958B5F53139AFC2Cull,
     0xA86295603F1E191Eull, 0x8480CD61D22A8FBAull, 0x43A96C6628C15A29ull,
     0x34B873D97739ECD7ull, 0xB4C67B1291D29335ull},
    {0x33DB5AE35E2362C5ull, 0x9096B83E963B98F4ull, 0x22CC6105ADA3D2F1ull,
     0xB9461755847AF081ull, 0xDB7C983985738991ull, 0xDC0742817B7611D2ull,
     0x3D3E514DEE637939ull, 0x4858FFD2058AFD50ull},
    {0xEFAEF1EFC38E2B0Bull, 0x5BA3FD1EF40E8003ull, 0x892088A8EEF73FE7ull,
     0x9065D092A464110Full, 0xD3EE0374494959ACull, 0x97D941B6805EF67Bull,
     0xFE996C9CC4C6CBA8ull, 0x4E69C3663BC36BFDull},
    {0xE5845E5BC866E0F9ull, 0x4FF8D90B74ED0202ull, 0x7D6ADA24719D279Eull,
     0x6CF94A3DC7CB39D3ull, 0x0ED4DA228C002F70ull, 0xBE96B97B207A6493ull,
     0xB910DB524F6898BAull, 0xAD69253AB0227950ull},
    {0x884B2C08E597FEF3ull, 0xF3F8E9524FE71C8Eull, 0xB42D20B4757F0C1Eull,
     0x38EB40B8E6E1DCE5ull, 0x2230DA1439EBC6B9ull, 0x7E27747E0E30102Full,
     0x9D870B3A4D44DA40ull, 0x2359EFA4CB2232E1ull},
    {0xDBD87149A2DEC22Aull, 0xD8D1942ADEB32F4Full, 0xDA079421DAB2FAB9ull,
     0x2432BA306246BC34ull, 0xF014E245DF0F5765ull, 0xA4ADBBA6A25BECCFull,
     0xF1105CE4CFA6FD1Cull, 0xDB9CF3B23D787901ull},
    {0x7F24B5FC76BEF3C3ull, 0xEEE9C8429F1D65ADull, 0xB64D3B7FC526AA6Dull,
     0x82FC8311CF2C6DD0ull, 0xBB90A2DA4C6E7EAEull, 0xE05A344262793DC3ull,
     0x3DB8D8C6C08F497Cull, 0xF0A297E3B18948C1ull},
    {0xD27959DDA5265763ull, 0x820BD1D40AEAA10Cull, 0xE619C237FED0DEB5ull,
     0x33FBCA21BBC4CC43ull, 0xF6F127609203610Eull, 0xE98C2A6E13967129ull,
     0x8EF8A62D36570048ull, 0x5F8FBA1323F7D039ull},
    {0x0FA78E79A8E055F6ull, 0xEC3C4773FB7ADCD4ull, 0x805A0F2FF5ABF71Eull,
     0xE14FB0D82910045Dull, 0x4ECBA4F2B61BA201ull, 0xFEEA2D880353A562ull,
     0x47C0015A9CB9F0A4ull, 0x1B4559517149C3C8ull},
    {0xD7CDE2E7751A8371ull, 0x1CCCE26BECF61B3Bull, 0x20E922C9159A8280ull,
     0xB0A9AF3AFC65CAFAull, 0x4D09614FDEEBD5F6ull, 0xDE625F5FB673BD78ull,
     0x0DDB3F1CAEC33FF1ull, 0x19BC796BCADC31B1ull},
    {0xCB680DB4D0DA4C70ull, 0xC37843D795CDAC7Full, 0x2DABEC059A28B405ull,
     0x06E4EB0CD504A592ull, 0x5478A70558AE90F2ull, 0xC951C7004A8BBFE5ull,
     0x6ABD09F5EEA05FD2ull, 0x75A7EA3FE5886620ull},
    {0x967A083AC902B253ull, 0xB36CF43137EA12C7ull, 0xDD57C31B38D4B188ull,
     0xF178D97A97707A4Bull, 0xA99813B0A5A7661Dull, 0x6537E5870C8E64F5ull,
     0xB5F65BA5084C7EC3ull, 0x9E2A88FC1113B525ull},
    {0x67AC5D67F2EE6254ull, 0x980A56BA43609F46ull, 0x9835244EC7AE1ECFull,
     0x96A5A75681B9830Eull, 0xA679C123E1768120ull, 0x4A44D42B11093842ull,
     0xD1E6D990EC69E4E3ull, 0xD135AB4F1E7E5EBEull},
    {0x972FD649FA988CB8ull, 0x629CC03E73D5A7AAull, 0xC130C3B105DA66DCull,
     0x896221DF1B25138Full, 0x8FBCFF1F626AB402ull, 0x8DAADD2E6561A365ull,
     0xC0CFCC9FD9A81F0Eull, 0x398A41FB70CFD960ull},
    {0xD23D5E66996965DCull, 0xFC37011D7CD3D670ull, 0x2C5B87C01942251Eull,
     0x7C9AC243F91D487Dull, 0x87C4D9552B9A7F66ull, 0x8889E1D2996B3AB5ull,
     0x6F009EBE32B506C1ull, 0x8D0CAF3CCD1A40D6ull},
    {0x60A597C1590A5D39ull, 0x1706D7E523750DC1ull, 0xE35D521DDB67E368ull,
     0x7063889406C19845ull, 0x693A17C5BF76DD30ull, 0xF3FD0D1DCA2FEC1Aull,
     0xA731D2D652981B20ull, 0xB31447FE32A34E6Eull},
    {0x03851ADAE16CE947ull, 0xAF8276A3ED350195ull, 0xDED3400F59A57B1Cull,
     0xBB35F7444FCC9402ull, 0xF47E9C240806C4E4ull, 0xF7F4E33D7084C6C0ull,
     0xDD8B651775AAA092ull, 0x0FBC186965ED1152ull},
    {0x79016CF83E698CCDull, 0x58DC97737CF373B9ull, 0xB3C8E2FC8CD7FB59ull,
     0x3F0C64A2F0D3BC56ull, 0x0EAB9AD23668D60Full, 0xA359A3742FC5C8C8ull,
     0x3E06F4A8A47AD2E9ull, 0x8B76134744D95D72ull},
    {0x9DCAB859FC62F018ull, 0xD53E8C247E85B3FAull, 0x74E2B05B9538EFD9ull,
     0x9545F2A75F75F430ull, 0x1F55558AC61FF5B8ull, 0xA6AAC9A2682A1F26ull,
     0x9BFD6B554D189ECCull, 0x523C8AC223536BD0ull},
    {0xBD393656A3CCA5A5ull, 0x29C962911D476A9Bull, 0xC6AF9D297583D693ull,
     0x29E30249334CD44Aull, 0x5498B68CF8471617ull, 0xF64B9070F7B58E33ull,
     0xD02436248C1185A5ull, 0x222AE8887F754EBEull},
    {0x28B26E4921B4350Full, 0xAF83B5A23366ECFFull, 0xD02A3CA9C512EC57ull,
     0x3CBA55BFC1ED0041ull, 0x1948E4F435C26CFCull, 0xD06F97AACE775989ull,
     0x54D9F58F776EB40Aull, 0xE88275706DBD1409ull},
    {0x51A221D849720A51ull, 0x27152DDE463CBAFEull, 0x51E1BEC0999B29E0ull,
     0x2367A82304F3F9E0ull, 0xA2D781E378426387ull, 0x8D5AE5DD0788C86Full,
     0x88F86AAA8B56A580ull, 0x6C103369467C5EB8ull},
    {0xC81C66E115EBADA9ull, 0xF33DCE56C7D52D73ull, 0xEC6157B80C50EE5Bull,
     0x2819A5842FBAAA87ull, 0x08B400BC9E54F651ull, 0x7E95B6FBFFE6A567ull,
     0x033420B7D100D004ull, 0x0B36A60483C400A5ull},
    {0x96DA68350079F75Aull, 0x2C7C7BC9799F41DBull, 0xC5A134CD399DBC55ull,
     0x50BB5EA2807BF7BCull, 0xB7250835A43C3DAEull, 0x426A92D622484DCDull,
     0xB3FC64E13FB64464ull, 0x5A5E5F0E1FDA2C25ull},
    {0xDB090F676B0F1225ull, 0x26786FF4745C67A4ull, 0xDFB7C66F47FB10EBull,
     0x1E4CA89D935A9D7Eull, 0x7C45A925F21E553Dull, 0x01FA686D7507D925ull,
     0xB7D1112239DEAAFEull, 0x34EBF39204C1A7AAull},
    {0x1245C88257FDB820ull, 0x2255DD4F7DAEA025ull, 0x6943EA9E56B060C1ull,
     0xF1B83C803DB0CA29ull, 0xCD3F576F6960A772ull, 0x6C1959E17E19F6E9ull,
     0x1D109A02E1D734E2ull, 0x253667FA1E7D265Full},
    {0x05085C5D2AD111E6ull, 0x2D6B26FD6CEA2516ull, 0x5A3C6BFF0636995Full,
     0xB6FC9E89200D0197ull, 0x5445DF41ECF73C93ull, 0x0A75908C178A7134ull,
     0xE911E4470117FCE5ull, 0xB55100F38558FB45ull},
    {0x694CC038FF4D1A94ull, 0x79C64D3F8D62F9F3ull, 0x4E523425F976AA37ull,
     0xD903711F40301619ull, 0x07FEE201CE9C788Bull, 0x52BAD1650341729Bull,
     0x6B80D9C426B659BEull, 0x8AB3C00D20133CBEull},
    {0xE09AAE6AA200CFE6ull, 0x76554546F3937079ull, 0x6EAED737F18D4189ull,
     0x0776BF3C41F9F38Aull, 0x864E9BA05361D19Eull, 0x438342DE62103910ull,
     0x819FE0ED6E60AB8Full, 0xBC4EB86C3786F15Cull},
    {0x89675DF8B86F87A9ull, 0x7891FBB0DEFA090Dull, 0x65CF768DD5EF1825ull,
     0x7D7DDBF29A70542Dull, 0x409000CC750F7C49ull, 0x79855E42F000C4D4ull,
     0x6634E5D0B8B58FF0ull, 0x70ACF2BD9CB37938ull},
    {0x65589D248E8B3AF4ull, 0xE089C6BFF8CB1AA3ull, 0x0CD9C369D3CE1DA4ull,
     0x2398BF0EDFE08AF2ull, 0x998EC01714C11B4Bull, 0x84994FA439FD9675ull,
     0xBB8CBAF6FEA0CA26ull, 0xD32C5E2D86F9F122ull},
    {0x27C6EC8DC82D9A3Cull, 0xDE417F2DFC33066Bull, 0xB29A6F42955DD9B2ull,
     0xDFFA60C3E608D38Full, 0x955F860A6ACB4BCDull, 0xC44B6E8F283F1A41ull,
     0x9546E75993ADB4ABull, 0xA7C0A1D393F81096ull},
    {0xB58D988F91B5C6B8ull, 0xA094DFABA33A9F56ull, 0x912F7892B945DF11ull,
     0x58D6CF410627C5A8ull, 0xB90FCDA2A67FF725ull, 0x56CED569E21EDD6Bull,
     0x9CE71F947B879804ull, 0x29DA623995A191A0ull},
    {0x487C9B3DEFCE9392ull, 0xCCE8DFD8D29BA49Aull, 0x0B5A3E68E51261B5ull,
     0xA653552E396A4F4Aull, 0x4BCAF0F0AF40EA60ull, 0x1FAB476065D1C953ull,
     0x27AC3D594488BF98ull, 0x29A2ADEF928B9015ull},
    {0x1654F0D16FB772BFull, 0xF4C17B9B90F9D4EBull, 0x06D2FB21F5FA24A2ull,
     0x3DE5C828173CB9D4ull, 0x5BF12A89D95781DEull, 0x35739DF32AA85209ull,
     0x9D55B6FD8A36B70Full, 0x1435B5CC87E6DAB3ull},
    {0x1C9FD58FCED51546ull, 0x67FE898149EDA3E3ull, 0x530DB4B45995406Dull,
     0x877E40701F913B72ull, 0xC64DBA28355B16B5ull, 0x0CB8F90C052BF3E1ull,
     0xBF5DFC336D9A8096ull, 0x630714B3A0B04F2Full},
    {0x445E307E616B3303ull, 0x5608336FE4032E3Bull, 0xDF95342140CD8735ull,
     0xF6EA6A5D0EECCDCCull, 0xCE0BE57362FCCDFEull, 0x7323EADEB9FDE253ull,
     0x733B0F9263DEF8FFull, 0x7CFA09B8015B49CFull},
    {0xB124A3BC4743EBE1ull, 0x3B8D4817D7E98903ull, 0x972CDEC9CEBFEAB6ull,
     0xD83B2C918B4DA9C5ull, 0x05721B687B01C8F4ull, 0x232EED1B6A4A5CF9ull,
     0x790A69D9B2436C69ull, 0xF14E7382C3841A4Bull},
    {0xEF789AD844F8AF14ull, 0xB48F7B0ED28354A1ull, 0x979306DED0432613ull,
     0x4CC25B45228A4B35ull, 0x35C82B54026E3699ull, 0x17DDE60F33288713ull,
     0x21647696F073E9B4ull, 0x6C413032878610DEull},
    {0xD23654A0CAFD8B2Cull, 0x11DA5AA13B5874BCull, 0x2FB00BDC193F7408ull,
     0xC2AF2D6B79F06540ull, 0x249C7F52C0BC1512ull, 0x83F8C3045A0409E7ull,
     0x67B641B7508BBAA1ull, 0xFE0A9F05B46E6605ull},
    {0x3B73A03EDA50F649ull, 0x72FA4263B59BA1ACull, 0xD10FF00672E124C1ull,
     0x9AA29A7A7EAF3C78ull, 0xA0A85EF2936ACADBull, 0xE6D271845E0BBA6Full,
     0x776CF0C1C36EBC1Full, 0xB8C66857A3AB5FA9ull},
    {0x06E5BB3A7D35E884ull, 0xB63BA7CE481FEC9Full, 0x3B3830309422E4B7ull,
     0xA6DFC65D6384904Aull, 0xAE7D37862494F6ABull, 0x3648E391FD7CF8D8ull,
     0x8A025B92FFE7EB86ull, 0x9A5E1574B45095D3ull},
    {0x0AC71D42F8B92095ull, 0x94DB3941EEC5650Bull, 0x72F89D7346ECFEABull,
     0xD8CC20A8152AB4CEull, 0x4725B25627068258ull, 0xDE5073EBCDE3AF10ull,
     0xACBDDD4503B79EF1ull, 0xC1F8ED375C00A3F9ull},
    {0x26CEE50776A249B3ull, 0x2DB99CD8E0903FAEull, 0x93ECE7C3A87A349Aull,
     0x3B8C4BF9D54012B2ull, 0x613426CD39751132ull, 0x3ACEB9CAE0F13AE6ull,
     0x1DDA81444FF991A3ull, 0x19467E5F596B154Eull},
    {0x500A4468C1B4833Aull, 0x10A07A387EFB912Full, 0x7CDB9305B4B6B1DCull,
     0xEAA8934F209C4F5Bull, 0xCA19CBAE7C82D98Eull, 0xDA4FEF5A14E5B820ull,
     0x2C704444DBD61AF7ull, 0x26BE706A60C836FAull},
    {0x542667AF2892FAACull, 0xA410F19544F13294ull, 0x6C4D99AD9ED8C096ull,
     0x1D09DE6080BFF316ull, 0x195927DF66C46703ull, 0xF744A832BED1E2BDull,
     0x24ABB84340EB5916ull, 0x0EA577B5C6EDE319ull},
    {0x13E4531090A79CA3ull, 0x837A91AC4C981704ull, 0xB9353228C87A4E01ull,
     0x7630BDF63AD675CDull, 0xAFC88227370C0304ull, 0xDEE89E41623C7672ull,
     0x1EC69273062E9BDBull, 0xD271191A0B31C6C6ull},
    {0xE43570AEF45EC0E9ull, 0x0C76C9418AE34328ull, 0xE48101BEC2FB5126ull,
     0x00DF38DCAB340E3Cull, 0x0979A1330EAD3B00ull, 0xF5AE287C0220E567ull,
     0xFD98921B6EA23666ull, 0x71B5E30FC68179DAull},
    {0x851322B322AC7990ull, 0x5689C1C4507FF169ull, 0xA3F012A45B2230FFull,
     0xF052A733DCAF73FFull, 0xFD830170171A8C8Dull, 0x568C4172080F4E5Cull,
     0x7FF9A21D828FF5EEull, 0xD35E973B9C09C0E9ull},
    {0x4402B7A90F47A250ull, 0x65122E648EDAC0D9ull, 0x848CA8A95405EECCull,
     0xCAA81AD486EF25C8ull, 0xBB11E91BE95E6118ull, 0xB562A8A76D725A2Cull,
     0x86D3CDF9CF6C5BAAull, 0x0999161B9D0D1079ull},
    {0x0F6D35A73CD22B28ull, 0xE1F7D3CF6C0F835Dull, 0x2DF2C75E25AB24DCull,
     0x5C290C545C0125AEull, 0x638C248891979519ull, 0xC2794335C3628205ull,
     0xC9140CD10A3DA435ull, 0xB326B45E28631794ull},
    {0xF21765E85E918760ull, 0x53F151CFF8019C86ull, 0xB4F1D7795D169533ull,
     0x71FEC2FE66039590ull, 0x5F737013EC97DF9Cull, 0x6FEFFBE0CF63268Aull,
     0x074A9DA74B335A3Bull, 0xDD89D23D091613C5ull},
    {0xFF55B36B0B1A7C2Eull, 0xCDFE45B6CAA12DADull, 0xAECBC2FA4156C75Aull,
     0xD7C267F26B3B1346ull, 0xA75E8798E6BA29E0ull, 0x042C2E4E1BD58E7Aull,
     0xC284FA62350406BFull, 0x9E9D58FCAC94843Dull},
    {0x1E27342B7FAB5D5Eull, 0x36651AABF2955B20ull, 0x466B0C8E641852B8ull,
     0xE4A55C2479B40173ull, 0x246C085F8F4A0EBBull, 0xFD6DAB30E0EA76E5ull,
     0xEEAD5E93B9527877ull, 0xD15B31A199AEC34Eull},
    {0x03A3EE3CBE75CDC8ull, 0x3EEF8A98B7617CB8ull, 0x5F3C4FDB1AD753C6ull,
     0xAD032BB04434C30Eull, 0x0717B27AF493E49Eull, 0x185791423952408Cull,
     0xF00CF4AD8BF88ABCull, 0x2AE204A20BC7956Cull},
    {0x9D86F42F4B98BC95ull, 0xCCD986DAC9E2D860ull, 0xFB77281FFDDF6F02ull,
     0x0211BBB10D42EEA7ull, 0x6BF82DCAA6726931ull, 0x64518B12D0EB6512ull,
     0x5DA3EC33A4FB1DD8ull, 0xAB3D158021C5F092ull},
    {0xA27A1F164B4FD923ull, 0xFBD465AD5E65EC8Eull, 0x1081815A8B2280AEull,
     0x2B7F950352224166ull, 0x6BDED9227D1D7488ull, 0x97ACC8A21CEAB2A3ull,
     0x6F71F1FD1120D7DCull, 0xAAA6A344BBDA97DAull},
    {0xA321F2819A46C446ull, 0x1DD32410D0C76676ull, 0xD0091407416CF82Aull,
     0x68A6F30D54A9A082ull, 0x7FA8A46C41B3148Aull, 0x696848E5463539B9ull,
     0x3C4F7F398A4B0C74ull, 0x75A7E639B9ACC245ull},
    {0xBAC7964535388D9Full, 0x95233FE6A7216DCEull, 0xF27F102F83C15D8Full,
     0xEEEDD0B5612D8B24ull, 0x2C50A1E311E65E4Cull, 0x3937FADC98DBD9A3ull,
     0xA93DAA798138FA12ull, 0x229B999F53B579B9ull},
    {0x488F69DFAD28E497ull, 0x2FF6234D55D62498ull, 0xD1603D8511382690ull,
     0x1847837742449FB9ull, 0x33B804ACAF82D97Cull, 0xD1BC0C2451F625C9ull,
     0x64BF1D658ADDAC5Cull, 0x8C9C1F59AA5DA6B1ull},
    {0xACF4EFA413E7481Bull, 0xF2DED502E55CA945ull, 0xF6518D7CD1D15F81ull,
     0x40B82CF4918F70E7ull, 0xE6711E953C621C60ull, 0xFBF300C59E8F0070ull,
     0xA0724ECB626742B6ull, 0x48AFD0E104CCB43Eull},
};

#endif  // PRINT_RESULTS

// Ensures Xorshift128+ returns consistent and unchanging values.
HWY_ATTR void TestGolden() {
  HWY_ALIGN Xorshift128Plus rng(12345);
  for (uint64_t vector = 0; vector < kVectors; ++vector) {
    HWY_ALIGN uint64_t lanes[Xorshift128Plus::N];
    rng.Fill(lanes);
#if PRINT_RESULTS
    Print(lanes);
#else
    for (size_t i = 0; i < Xorshift128Plus::N; ++i) {
      ASSERT_EQ(kExpected[vector][i], lanes[i])
          << "Where vector=" << vector << " i=" << i;
    }
#endif
  }
}

// Output changes when given different seeds
HWY_ATTR void TestSeedChanges() {
  HWY_ALIGN uint64_t lanes[Xorshift128Plus::N];

  std::vector<uint64_t> first;
  constexpr size_t kNumSeeds = 16384;
  first.reserve(kNumSeeds);

  // All 14-bit seeds
  for (size_t seed = 0; seed < kNumSeeds; ++seed) {
    HWY_ALIGN Xorshift128Plus rng(seed);

    rng.Fill(lanes);
    first.push_back(lanes[0]);
  }

  // All outputs are unique
  ASSERT_EQ(kNumSeeds, first.size());
  std::sort(first.begin(), first.end());
  first.erase(std::unique(first.begin(), first.end()), first.end());
  EXPECT_EQ(kNumSeeds, first.size());
}

HWY_ATTR void TestFloat() {
  ThreadPoolInternal pool(8);

  // All 14-bit seeds
  pool.Run(0, 16384, ThreadPool::SkipInit(),
           [](const int seed, const int /*thread*/) HWY_ATTR {
             HWY_ALIGN Xorshift128Plus rng(seed);

             const HWY_FULL(uint32_t) du;
             const HWY_FULL(float) df;
             HWY_ALIGN uint64_t batch[Xorshift128Plus::N];
             HWY_ALIGN float lanes[df.N];
             double sum = 0.0;
             size_t count = 0;
             const size_t kReps = 2000;
             for (size_t reps = 0; reps < kReps; ++reps) {
               rng.Fill(batch);
               for (size_t i = 0; i < Xorshift128Plus::N * 2; i += df.N) {
                 const auto bits =
                     Load(du, reinterpret_cast<const uint32_t*>(batch) + i);
                 // 1.0 + 23 random mantissa bits = [1, 2)
                 const auto rand12 = BitCast(
                     df, hwy::ShiftRight<9>(bits) | Set(du, 0x3F800000));
                 const auto rand01 = rand12 - Set(df, 1.0f);
                 Store(rand01, df, lanes);
                 for (float lane : lanes) {
                   sum += lane;
                   count += 1;
                   EXPECT_LE(lane, 1.0f);
                   EXPECT_GE(lane, 0.0f);
                 }
               }
             }

             // Verify average (uniform distribution)
             EXPECT_NEAR(0.5, sum / count, 0.00702);
           });
}

// Not more than one 64-bit zero
HWY_ATTR void TestNotZero() {
  ThreadPoolInternal pool(8);

  pool.Run(0, 2000, ThreadPool::SkipInit(),
           [](const int task, const int /*thread*/) HWY_ATTR {
             HWY_ALIGN uint64_t lanes[Xorshift128Plus::N];

             HWY_ALIGN Xorshift128Plus rng(task);
             size_t num_zero = 0;
             for (size_t vectors = 0; vectors < 10000; ++vectors) {
               rng.Fill(lanes);
               for (uint64_t lane : lanes) {
                 num_zero += static_cast<size_t>(lane == 0);
               }
             }
             EXPECT_LE(num_zero, 1);
           });
}

HWY_ATTR HWY_NOINLINE void RunAll() {
  TestNotZero();
  TestGolden();
  TestSeedChanges();
  TestFloat();
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl

// Instantiate for the current target.
void XorshiftTest::HWY_FUNC() { jxl::HWY_NAMESPACE::RunAll(); }