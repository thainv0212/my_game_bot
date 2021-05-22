class ActionMap:
    actionMap = {}

    def __init__(self):
        actions = 'AIR AIR_A AIR_B AIR_D_DB_BA AIR_D_DB_BB AIR_D_DF_FA AIR_D_DF_FB AIR_DA AIR_DB AIR_F_D_DFA AIR_F_D_DFB AIR_FA AIR_FB AIR_GUARD AIR_GUARD_RECOV AIR_RECOV AIR_UA AIR_UB BACK_JUMP BACK_STEP CHANGE_DOWN CROUCH CROUCH_A CROUCH_B CROUCH_FA CROUCH_FB CROUCH_GUARD CROUCH_GUARD_RECOV CROUCH_RECOV DASH DOWN FOR_JUMP FORWARD_WALK JUMP LANDING NEUTRAL RISE STAND STAND_A STAND_B STAND_D_DB_BA STAND_D_DB_BB STAND_D_DF_FA STAND_D_DF_FB STAND_D_DF_FC STAND_F_D_DFA STAND_F_D_DFB STAND_FA STAND_FB STAND_GUARD STAND_GUARD_RECOV STAND_RECOV THROW_A THROW_B THROW_HIT THROW_SUFFER'
        actions = actions.split(' ')
        for i, action in enumerate(actions):
            self.actionMap[i] = action
        # self.actionMap[0] = "FORWARD_WALK"
        # self.actionMap[1] = "DASH"
        # self.actionMap[2] = "BACK_STEP"
        # self.actionMap[3] = "JUMP"
        # self.actionMap[4] = "FOR_JUMP"
        # self.actionMap[5] = "BACK_JUMP"
        # self.actionMap[6] = "STAND_GUARD"
        # self.actionMap[7] = "CROUCH_GUARD"
        # self.actionMap[8] = "AIR_GUARD"
        # self.actionMap[9] = "THROW_A"
        # self.actionMap[10] = "THROW_B"
        # self.actionMap[11] = "STAND_A"
        # self.actionMap[12] = "STAND_B"
        # self.actionMap[13] = "CROUCH_A"
        # self.actionMap[14] = "CROUCH_B"
        # self.actionMap[15] = "AIR_A"
        # self.actionMap[16] = "AIR_B"
        # self.actionMap[17] = "AIR_DA"
        # self.actionMap[18] = "AIR_DB"
        # self.actionMap[19] = "STAND_FA"
        # self.actionMap[20] = "STAND_FB"
        # self.actionMap[21] = "CROUCH_FA"
        # self.actionMap[22] = "CROUCH_FB"
        # self.actionMap[23] = "AIR_FA"
        # self.actionMap[24] = "AIR_FB"
        # self.actionMap[25] = "AIR_UA"
        # self.actionMap[26] = "AIR_UB"
        # self.actionMap[27] = "STAND_D_DF_FA"
        # self.actionMap[28] = "STAND_D_DF_FB"
        # self.actionMap[29] = "STAND_F_D_DFA"
        # self.actionMap[30] = "STAND_F_D_DFB"
        # self.actionMap[31] = "STAND_D_DB_BA"
        # self.actionMap[32] = "STAND_D_DB_BB"
        # self.actionMap[33] = "AIR_D_DF_FA"
        # self.actionMap[34] = "AIR_D_DF_FB"
        # self.actionMap[35] = "AIR_F_D_DFA"
        # self.actionMap[36] = "AIR_F_D_DFB"
        # self.actionMap[37] = "AIR_D_DB_BA"
        # self.actionMap[38] = "AIR_D_DB_BB"
        # self.actionMap[39] = "STAND_D_DF_FC"


ACTIONS = ["FORWARD_WALK", "DASH", "BACK_STEP", "JUMP", "FOR_JUMP", "BACK_JUMP", "STAND_GUARD", "CROUCH_GUARD",
           "AIR_GUARD", "THROW_A", "THROW_B", "STAND_A", "STAND_B", "CROUCH_A", "CROUCH_B", "AIR_A", "AIR_B", "AIR_DA",
           "AIR_DB", "STAND_FA", "STAND_FB", "CROUCH_FA", "CROUCH_FB", "AIR_FA", "AIR_FB", "AIR_UA", "AIR_UB",
           "STAND_D_DF_FA", "STAND_D_DF_FB", "STAND_F_D_DFA", "STAND_F_D_DFB", "STAND_D_DB_BA", "STAND_D_DB_BB",
           "AIR_D_DF_FA", "AIR_D_DF_FB", "AIR_F_D_DFA", "AIR_F_D_DFB", "AIR_D_DB_BA", "AIR_D_DB_BB", "STAND_D_DF_FC"]
