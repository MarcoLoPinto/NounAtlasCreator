import os
import numpy as np

import json
from collections import Counter

from copy import deepcopy

from transformers import AutoTokenizer
import torch
from random import shuffle

class DatasetNoUniteD():
    def __init__(self, lang_data_path:str, split_type_to_use = '', split_predicates = False, max_length = None, shuffle = False):

        self.nolabel_value = '_'
        self.split_predicates = split_predicates
        self.max_length = max_length
        self.shuffle = shuffle
        self.split_type_to_use = split_type_to_use # either "_v" (only verbal), "_n" (only nominal) or "" (all)
        
        # adding roles
        self.id_to_roles = [self.nolabel_value] + [
            'agent', 'asset', 'attribute', 'beneficiary', 'cause', 'co-agent', 'co-patient', 
            'co-theme', 'destination', 'experiencer', 'extent', 'goal', 'idiom', 'instrument', 'location', 
            'material', 'patient', 'product', 'purpose', 'recipient', 'result', 'source', 'stimulus', 
            'theme', 'time', 'topic', 'value']
        self.roles_to_id = {role:i for i,role in enumerate(self.id_to_roles)}
        self.roles_pad_id = -1
        self.roles_pad = self.id_to_roles[self.roles_pad_id]

        # adding predicates (= frames)
        self.id_to_predicates = [self.nolabel_value] + [
            'ABSORB', 'ABSTAIN_AVOID_REFRAIN', 'ACCOMPANY', 'ACCUSE', 'ACHIEVE', 'ADD', 'ADJUST_CORRECT', 'AFFECT', 
            'AFFIRM', 'AGREE_ACCEPT', 'AIR', 'ALLY_ASSOCIATE_MARRY', 'ALTERNATE', 'AMASS', 'AMELIORATE', 'ANALYZE', 'ANSWER', 
            'APPEAR', 'APPLY', 'APPROVE_PRAISE', 'ARGUE-IN-DEFENSE', 'AROUSE_WAKE_ENLIVEN', 'ARRIVE', 'ASCRIBE', 'ASK_REQUEST', 
            'ASSIGN-SMT-TO-SMN', 'ATTACH', 'ATTACK_BOMB', 'ATTEND', 'ATTRACT_SUCK', 'AUTHORIZE_ADMIT', 'AUTOMATIZE', 'AUXILIARY', 
            'AUX_MOD', 'BE-LOCATED_BASE', 'BEFRIEND', 'BEGIN', 'BEHAVE', 'BELIEVE', 'BEND', 'BENEFIT_EXPLOIT', 'BETRAY', 'BEWITCH', 
            'BID', 'BLIND', 'BORDER', 'BREAK_DETERIORATE', 'BREATH_BLOW', 'BRING', 'BULGE-OUT', 'BURDEN_BEAR', 'BURN', 'BURY_PLANT', 
            'BUY', 'CAGE_IMPRISON', 'CALCULATE_ESTIMATE', 'CANCEL_ELIMINATE', 'CARRY-OUT-ACTION', 'CARRY_TRANSPORT', 'CASTRATE', 'CATCH', 
            'CATCH_EMBARK', 'CAUSE-MENTAL-STATE', 'CAUSE-SMT', 'CAVE_CARVE', 'CELEBRATE_PARTY', 'CHANGE-APPEARANCE/STATE', 'CHANGE-HANDS', 
            'CHANGE-TASTE', 'CHANGE_SWITCH', 'CHARGE', 'CHASE', 'CHOOSE', 'CIRCULATE_SPREAD_DISTRIBUTE', 'CITE', 'CLOSE', 'CLOUD_SHADOW_HIDE', 
            'CO-OPT', 'COLOR', 'COMBINE_MIX_UNITE', 'COME-AFTER_FOLLOW-IN-TIME', 'COME-FROM', 'COMMUNE', 'COMMUNICATE_CONTACT', 'COMMUNIZE', 
            'COMPARE', 'COMPENSATE', 'COMPETE', 'COMPLEXIFY', 'CONQUER', 'CONSIDER', 'CONSUME_SPEND', 'CONTAIN', 'CONTINUE', 
            'CONTRACT-AN-ILLNESS_INFECT', 'CONVERT', 'COOK', 'COOL', 'COPULA', 'COPY', 'CORRELATE', 'CORRODE_WEAR-AWAY_SCRATCH', 'CORRUPT', 'COST', 
            'COUNT', 'COURT', 'COVER_SPREAD_SURMOUNT', 'CREATE_MATERIALIZE', 'CRITICIZE', 'CRY', 'CUT', 'DANCE', 'DEBASE_ADULTERATE', 'DECEIVE', 
            'DECIDE_DETERMINE', 'DECREE_DECLARE', 'DEFEAT', 'DELAY', 'DERIVE', 'DESTROY', 'DEVELOP_AGE', 'DIET', 'DIM', 'DIP_DIVE', 'DIRECT_AIM_MANEUVER', 
            'DIRTY', 'DISAPPEAR', 'DISBAND_BREAK-UP', 'DISCARD', 'DISCOURSE-FUNCTION', 'DISCUSS', 'DISLIKE', 'DISMISS_FIRE-SMN', 'DISTINGUISH_DIFFER', 
            'DIVERSIFY', 'DIVIDE', 'DOWNPLAY_HUMILIATE', 'DRESS_WEAR', 'DRINK', 'DRIVE-BACK', 'DROP', 'DRY', 'EARN', 'EAT_BITE', 'EMBELLISH', 'EMCEE', 
            'EMIT', 'EMPHASIZE', 'EMPTY_UNLOAD', 'ENCLOSE_WRAP', 'ENDANGER', 'ENJOY', 'ENTER', 'ESTABLISH', 'EXCRETE', 'EXEMPT', 'EXHAUST', 'EXIST-WITH-FEATURE', 
            'EXIST_LIVE', 'EXPLAIN', 'EXPLODE', 'EXTEND', 'EXTRACT', 'FACE_CHALLENGE', 'FACIAL-EXPRESSION', 'FAIL_LOSE', 'FAKE', 'FALL_SLIDE-DOWN', 'FEEL', 
            'FIGHT', 'FILL', 'FIND', 'FINISH_CONCLUDE_END', 'FIT', 'FLATTEN_SMOOTHEN', 'FLATTER', 'FLOW', 'FLY', 'FOCUS', 'FOLLOW-IN-SPACE', 
            'FOLLOW_SUPPORT_SPONSOR_FUND', 'FORGET', 'FRUSTRATE_DISAPPOINT', 'FUEL', 'GENERATE', 'GIVE-BIRTH', 'GIVE-UP_ABOLISH_ABANDON', 'GIVE_GIFT', 
            'GO-FORWARD', 'GROUND_BASE_FOUND', 'GROUP', 'GROW_PLOW', 'GUARANTEE_ENSURE_PROMISE', 'GUESS', 'HANG', 'HAPPEN_OCCUR', 'HARMONIZE', 
            'HAVE-A-FUNCTION_SERVE', 'HAVE-SEX', 'HEAR_LISTEN', 'HEAT', 'HELP_HEAL_CARE_CURE', 'HIRE', 'HIT', 'HOLE_PIERCE', 'HOST_MEAL_INVITE', 'HUNT', 
            'HURT_HARM_ACHE', 'IMAGINE', 'IMPLY', 'INCITE_INDUCE', 'INCLINE', 'INCLUDE-AS', 'INCREASE_ENLARGE_MULTIPLY', 'INFER', 'INFLUENCE', 'INFORM', 'INSERT', 
            'INTERPRET', 'INVERT_REVERSE', 'ISOLATE', 'JOIN_CONNECT', 'JOKE', 'JUMP', 'JUSTIFY_EXCUSE', 'KILL', 'KNOCK-DOWN', 'KNOW', 'LAND_GET-OFF', 'LAUGH', 
            'LEAD_GOVERN', 'LEARN', 'LEAVE-BEHIND', 'LEAVE_DEPART_RUN-AWAY', 'LEND', 'LIBERATE_ALLOW_AFFORD', 'LIE', 'LIGHT-VERB', 'LIGHTEN', 'LIGHT_SHINE', 
            'LIKE', 'LOAD_PROVIDE_CHARGE_FURNISH', 'LOCATE-IN-TIME_DATE', 'LOSE', 'LOWER', 'LURE_ENTICE', 'MAKE-A-SOUND', 'MAKE-RELAX', 'MANAGE', 'MATCH', 
            'MEAN', 'MEASURE_EVALUATE', 'MEET', 'MESS', 'METEOROLOGICAL', 'MISS_OMIT_LACK', 'MISTAKE', 'MODAL', 'MOUNT_ASSEMBLE_PRODUCE', 'MOVE-BACK', 
            'MOVE-BY-MEANS-OF', 'MOVE-ONESELF', 'MOVE-SOMETHING', 'MUST', 'NAME', 'NEGOTIATE', 'NOURISH_FEED', 'OBEY', 'OBLIGE_FORCE', 'OBTAIN', 'ODORIZE', 
            'OFFEND_DISESTEEM', 'OFFER', 'OPEN', 'OPERATE', 'OPPOSE_REBEL_DISSENT', 'ORDER', 'ORGANIZE', 'ORIENT', 'OVERCOME_SURPASS', 'OVERLAP', 'PAINT', 
            'PARDON', 'PARTICIPATE', 'PAY', 'PERCEIVE', 'PERFORM', 'PERMEATE', 'PERSUADE', 'PLAN_SCHEDULE', 'PLAY_SPORT/GAME', 'POPULATE', 'POSSESS', 'PRECEDE', 
            'PRECLUDE_FORBID_EXPEL', 'PREPARE', 'PRESERVE', 'PRESS_PUSH_FOLD', 'PRETEND', 'PRINT', 'PROMOTE', 'PRONOUNCE', 'PROPOSE', 'PROTECT', 'PROVE', 
            'PUBLICIZE', 'PUBLISH', 'PULL', 'PUNISH', 'PUT_APPLY_PLACE_PAVE', 'QUARREL_POLEMICIZE', 'RAISE', 'REACH', 'REACT', 'READ', 'RECALL', 'RECEIVE', 
            'RECOGNIZE_ADMIT_IDENTIFY', 'RECORD', 'REDUCE_DIMINISH', 'REFER', 'REFLECT', 'REFUSE', 'REGRET_SORRY', 'RELY', 'REMAIN', 'REMEMBER', 
            'REMOVE_TAKE-AWAY_KIDNAP', 'RENEW', 'REPAIR_REMEDY', 'REPEAT', 'REPLACE', 'REPRESENT', 'REPRIMAND', 'REQUIRE_NEED_WANT_HOPE', 'RESERVE', 
            'RESIGN_RETIRE', 'RESIST', 'REST', 'RESTORE-TO-PREVIOUS/INITIAL-STATE_UNDO_UNWIND', 'RESTRAIN', 'RESULT_CONSEQUENCE', 'RETAIN_KEEP_SAVE-MONEY', 
            'REVEAL', 'RISK', 'ROLL', 'RUN', 'SATISFY_FULFILL', 'SCORE', 'SEARCH', 'SECURE_FASTEN_TIE', 'SEE', 'SEEM', 'SELL', 'SEND', 'SEPARATE_FILTER_DETACH', 
            'SETTLE_CONCILIATE', 'SEW', 'SHAPE', 'SHARE', 'SHARPEN', 'SHOOT_LAUNCH_PROPEL', 'SHOUT', 'SHOW', 'SIGN', 'SIGNAL_INDICATE', 'SIMPLIFY', 'SIMULATE', 
            'SING', 'SLEEP', 'SLOW-DOWN', 'SMELL', 'SOLVE', 'SORT_CLASSIFY_ARRANGE', 'SPEAK', 'SPEED-UP', 'SPEND-TIME_PASS-TIME', 'SPILL_POUR', 'SPOIL', 
            'STABILIZE_SUPPORT-PHYSICALLY', 'START-FUNCTIONING', 'STAY_DWELL', 'STEAL_DEPRIVE', 'STOP', 'STRAIGHTEN', 'STRENGTHEN_MAKE-RESISTANT', 'STUDY', 
            'SUBJECTIVE-JUDGING', 'SUBJUGATE', 'SUMMARIZE', 'SUMMON', 'SUPPOSE', 'SWITCH-OFF_TURN-OFF_SHUT-DOWN', 'TAKE', 'TAKE-A-SERVICE_RENT', 
            'TAKE-INTO-ACCOUNT_CONSIDER', 'TAKE-SHELTER', 'TASTE', 'TEACH', 'THINK', 'THROW', 'TIGHTEN', 'TOLERATE', 'TOUCH', 'TRANSLATE', 'TRANSMIT', 'TRAVEL', 
            'TREAT', 'TREAT-WITH/BY', 'TRY', 'TURN_CHANGE-DIRECTION', 'TYPE', 'UNDERGO-EXPERIENCE', 'UNDERSTAND', 'UNFASTEN_UNFOLD', 'USE', 'VERIFY', 'VIOLATE', 
            'VISIT', 'WAIT', 'WARN', 'WASH_CLEAN', 'WASTE', 'WATCH_LOOK-OUT', 'WEAKEN', 'WEAVE', 'WELCOME', 'WET', 'WIN', 'WORK', 'WORSEN', 'WRITE']
        self.predicates_to_id = {pred:i for i,pred in enumerate(self.id_to_predicates)}
        self.predicates_pad_id = -1
        self.predicates_pad = self.id_to_predicates[self.predicates_pad_id]

        self.data = self.load_data(lang_data_path)

    def create_collate_fn(self):
        def collate_fn(batch):
            batch_formatted = {}

            batch_formatted['words'] = [sample['words'] for sample in batch]
            
            batch_formatted['predicates'] = [[p.upper() for p in sample['predicates']] for sample in batch]
            batch_formatted['predicates_positions'] = [[1 if s != self.nolabel_value else 0 for s in predicates] for predicates in batch_formatted['predicates']]

            if self.split_predicates:
                batch_formatted['predicate_word'] = [sample['predicate_word'] for sample in batch]
                batch_formatted['predicate_name'] = [[p.upper() for p in sample['predicate_name']] for sample in batch]
                batch_formatted['roles'] = [[r.lower() for r in sample['roles']] for sample in batch]
                batch_formatted['predicate_position'] = [sample['predicate_position'] for sample in batch]

            return batch_formatted
        return collate_fn

    def load_data(self, data_path):

        with open(data_path) as json_file:
            d = json.load(json_file)
        d = list(d.values()) if type(d) == dict else d

        if self.split_predicates:

            d_formatted = []
            predicates_type = f'predicates{self.split_type_to_use}'
            roles_type = f'roles{self.split_type_to_use}'

            for sample in d:
                if not all(p == '_' for p in sample[predicates_type]): # at least one predicate, so to have roles for the AIC part:
                    for i, predicate in enumerate(sample[predicates_type]):
                        if predicate == '_':
                            continue
                        sample_copy = deepcopy(sample)
                        preds = sample_copy[predicates_type]
                        sample_copy['predicates'] = ['_']*i + [preds[i]] + ['_']*(len(preds)-i-1) # removing every other predicate in the phrase
                        sample_copy['predicate_word'] = [sample['words'][i]]
                        sample_copy['predicate_name'] = [preds[i]]
                        sample_copy['predicate_position'] = [i]
                        
                        # get the roles for that particular predicate as list! (if it has the "roles" attribute)
                        if roles_type in sample_copy and type(sample_copy[roles_type]) == dict and len(sample_copy[roles_type]) > 0:
                            try:
                                roles = sample_copy[roles_type][int(i)]
                            except KeyError:
                                try:
                                    roles = sample_copy[roles_type][str(i)]
                                except:
                                    roles = []
                            except:
                                roles = []
                            sample_copy['roles'] = roles
                        else:
                            sample_copy['roles'] = []

                        d_formatted.append(sample_copy)

            d = d_formatted[:self.max_length]
            
        else:
            d = d[:self.max_length]

        if self.shuffle: shuffle(d)

        return d

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]