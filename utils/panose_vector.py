import os
from fontTools.ttLib import TTFont, TTCollection
from unicodedata import normalize

PATH_FONTS = os.path.join(os.path.dirname(__file__), '..', 'fonts')
SIZE = 17

class PanoseVector:

    @staticmethod
    def extended_vector(font):
        vector = [-1] * SIZE
        panose = font['OS/2'].panose
        vector[0] = panose.bFamilyType # инд
        vector[1] = panose.bWeight
        vector[2] = panose.bContrast

        if vector[0] == 2 or vector[0] == 0:
            vector[3] = panose.bSerifStyle # инд
            vector[4] = panose.bProportion
            vector[5] = panose.bStrokeVariation # инд
            vector[6] = panose.bArmStyle # инд
            vector[7] = panose.bLetterForm # инд
            vector[8] = panose.bMidline
            vector[9] = panose.bXHeight

        elif vector[0] == 3:
            vector[10] = panose.bSerifStyle # ToolKind - инд
            vector[11] = panose.bProportion # Spacing - инд
            vector[12] = panose.bStrokeVariation # AspectRatio
            vector[13] = panose.bArmStyle # Topology - инд
            vector[14] = panose.bLetterForm # Form - инд
            vector[15] = panose.bMidline # Finials - инд
            vector[16] = panose.bXHeight # XAscent
        return vector

    @staticmethod
    def normalize(vector):
        normalized = []

        # Family Type
        family = [0] * 6
        family[vector[0]] = 1
        normalized.extend(family)

        # Weight
        normalized.append(round(vector[1] / 11, 2) if vector[1] > 0 else 0)
        # Contrast
        normalized.append(round(vector[2] / 9, 2) if vector[2] > 0 else 0)

        # Serif Style
        serif = [0] * 16
        if 0 <= vector[3] < 16:
            serif[vector[3]] = 1
        normalized.extend(serif)
        # Proportion
        normalized.append(round(vector[4] / 9, 2) if vector[4] > 0 else 0)
        # Stroke Variation
        stroke_variation = [0] * 11
        if 0 <= vector[5] < 11:
            stroke_variation[vector[5]] = 1
        normalized.extend(stroke_variation)
        # Arm Style
        arm_style = [0] * 11
        if 0 <= vector[6] < 11:
            arm_style[vector[6]] = 1
        normalized.extend(arm_style)
        # Letter Form
        letter_form = [0] * 16
        if 0 <= vector[7] < 16:
            letter_form[vector[7]] = 1
        normalized.extend(letter_form)
        # Midline
        normalized.append(round(vector[8] / 13, 2) if vector[8] > 0 else 0)
        # X-height
        normalized.append(round(vector[9] / 7, 2) if vector[9] > 0 else 0)

        #Tool Kind
        tool_kind = [0] * 10
        if 0 <= vector[10] < 10:
            tool_kind[vector[10]] = 1
        normalized.extend(tool_kind)
        # Spacing
        spacing = [0] * 4
        if 0 <= vector[11] < 4:
            spacing[vector[11]] = 1
        normalized.extend(spacing)
        #Aspect Ratio
        normalized.append(round(vector[12] / 6, 2) if vector[12] > 0 else 0)
        # Topology
        topology = [0] * 11
        if 0 <= vector[13] < 11:
            topology[vector[13]] = 1
        normalized.extend(topology)
        # Form
        form = [0] * 14
        if 0 <= vector[14] < 14:
            form[vector[14]] = 1
        normalized.extend(form)
        # Finials
        finials = [0] * 14
        if 0 <= vector[15] < 14:
            finials[vector[15]] = 1
        normalized.extend(finials)
        # X-ascent
        normalized.append(round(vector[16] / 6, 2) if vector[16] > 0 else 0)

        return normalized


    @staticmethod
    def get_vector():
        result = {}
        def process_font(font, name):
            vec = PanoseVector.extended_vector(font)
            norm_vec = PanoseVector.normalize(vec)
            # print(name)
            # print(vec)
            # print(norm_vec)
            # print(len(norm_vec))
            result[name] = norm_vec

        for file in os.listdir(PATH_FONTS):
            path = os.path.join(PATH_FONTS, file)
            if file.endswith('.ttc'):
                ttc = TTCollection(path)
                for i, font in enumerate(ttc.fonts):
                    font_name = f"{os.path.splitext(file)[0]}_{i}"
                    process_font(font, font_name)
            else:
                font = TTFont(path)
                font_name = os.path.splitext(file)[0]
                process_font(font, font_name)

        return result

a = PanoseVector.get_vector()
print(a)