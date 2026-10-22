import unittest
from match_state import TeamMemory

ROSTER={'1744X':'red','471A':'red','88S':'blue','2899A':'blue'}


class ReconnectTests(unittest.TestCase):
    def initial(self):
        return [dict(team=t,track_id=i,xyxy=[.1+.2*i,.2,.2+.2*i,.3],confidence=.9)
                for i,t in enumerate(ROSTER)]

    def test_new_nearby_track_recovers_only_missing_identity(self):
        m=TeamMemory(ROSTER);first=self.initial();m.update(0,first)
        current=[dict(o) for o in first if o['team']!='88S']
        current.append(dict(team=None,track_id=66,xyxy=[.51,.2,.61,.3]))
        states=m.update(.3,current)
        self.assertEqual(current[-1]['team'],'88S')
        self.assertEqual(current[-1]['alliance'],'blue')
        self.assertEqual(current[-1]['identity_source'],'recent_lost_spatial_reconnect')
        self.assertEqual(states[2]['track_id'],66)
        self.assertEqual(len({o['team'] for o in current}),4)

    def test_no_assignment_when_two_teams_missing_or_peer_changed(self):
        for change_peer in (False,True):
            m=TeamMemory(ROSTER);first=self.initial();m.update(0,first)
            current=[dict(o) for o in first if o['team']!='88S']
            if change_peer:current[0]['track_id']=99
            else:current.pop(0)
            current.append(dict(team=None,track_id=66,xyxy=[.51,.2,.61,.3]))
            m.update(.3,current);self.assertIsNone(current[-1]['team'])

    def test_no_reconnection_far_late_ambiguous_or_reviewed_uncertain(self):
        for case in ('far','late','ambiguous','review'):
            m=TeamMemory(ROSTER);first=self.initial();m.update(0,first)
            current=[dict(o) for o in first if o['team']!='88S']
            unknown=dict(team=None,track_id=66,xyxy=[.51,.2,.61,.3])
            if case=='far':unknown['xyxy']=[.1,.7,.2,.8]
            if case=='review':unknown['identity_source']='reviewed_overlap_uncertain'
            if case=='ambiguous':current.append(dict(team=None,track_id=67,xyxy=[.52,.2,.62,.3]))
            current.append(unknown)
            m.update(3 if case=='late' else .3,current)
            self.assertIsNone(unknown['team'])

    def test_spatial_recovery_does_not_steal_a_duplicate_box_of_known_peer(self):
        memory=TeamMemory(ROSTER);first=self.initial()
        first[-1]['xyxy']=[.60,.2,.70,.3]
        memory.update(0,first)
        current=[dict(o) for o in first if o['team']!='88S']
        # Close enough to the lost 88S for the old spatial rule, but contained
        # in the independently visible 2899A. It must not receive 88S's team.
        unknown=dict(team=None,track_id=66,xyxy=[.605,.205,.655,.295],confidence=.9)
        current.append(unknown)
        memory.update(.3,current)
        self.assertIsNone(unknown['team'])

    def test_spatial_recovery_rejects_outside_field_and_pin_candidates(self):
        for extra in (dict(field_validated=False),dict(class_id=9)):
            memory=TeamMemory(ROSTER);first=self.initial();memory.update(0,first)
            current=[dict(o) for o in first if o['team']!='88S']
            unknown=dict(team=None,track_id=66,xyxy=[.51,.2,.61,.3],confidence=.9,**extra)
            current.append(unknown);memory.update(.3,current)
            self.assertIsNone(unknown['team'])

    def stable_peers_after_long_occlusion(self):
        memory=TeamMemory(ROSTER);first=self.initial();memory.update(0,first)
        peers=[dict(o) for o in first if o['team']!='88S']
        for t in (.4,.8,1.2,1.6,2.,2.4,2.8,3.2):
            memory.update(t,[dict(o) for o in peers])
        return memory,peers

    def test_unique_fourth_recovers_after_long_occlusion_and_large_move(self):
        memory,peers=self.stable_peers_after_long_occlusion()
        for t in (3.25,3.30,3.35):
            unknown=dict(team=None,track_id=66,xyxy=[.1,.7,.2,.8],confidence=.8,field_validated=True)
            current=[dict(o) for o in peers]+[unknown]
            memory.update(t,current)
            self.assertEqual(len(current),4)
            if t<3.35:self.assertIsNone(unknown['team'])
        self.assertEqual(unknown['team'],'88S')
        self.assertEqual(unknown['identity_source'],'unique_remaining_robot_reconnect')
        self.assertEqual(unknown['xyxy'],[.1,.7,.2,.8])

    def test_fourth_fallback_rejects_ambiguous_or_non_robot_evidence(self):
        for case in ('duplicate','outside','weak','pin','two_unknown','changed_peer','blocked_review'):
            with self.subTest(case=case):
                memory,peers=self.stable_peers_after_long_occlusion()
                for t in (3.25,3.30,3.35,3.40):
                    current=[dict(o) for o in peers]
                    unknown=dict(team=None,track_id=66,xyxy=[.1,.7,.2,.8],confidence=.8,field_validated=True)
                    if case=='duplicate':unknown['xyxy']=[.11,.21,.19,.29]
                    if case=='outside':unknown['field_validated']=False
                    if case=='weak':unknown['confidence']=.2
                    if case=='pin':unknown['class_id']=9
                    if case=='two_unknown':current[0]['team']=None
                    if case=='changed_peer':current[0]['track_id']=int(t*100)
                    if case=='blocked_review':unknown['identity_source']='reviewed_overlap_uncertain'
                    current.append(unknown);memory.update(t,current)
                    self.assertIsNone(unknown['team'])

    def test_fourth_requires_continuity_after_one_frame_false_positive(self):
        memory,peers=self.stable_peers_after_long_occlusion()
        def current():
            return [dict(o) for o in peers]+[dict(team=None,track_id=66,xyxy=[.1,.7,.2,.8],confidence=.8)]
        first=current();memory.update(3.25,first)
        memory.update(3.30,[dict(o) for o in peers])
        resumed=current();memory.update(3.35,resumed)
        self.assertIsNone(resumed[-1]['team'])
        jumped=current();jumped[-1]['xyxy']=[.7,.7,.8,.8];memory.update(3.4,jumped)
        self.assertIsNone(jumped[-1]['team'])


if __name__=='__main__':unittest.main()
