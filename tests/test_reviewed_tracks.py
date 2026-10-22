import unittest
from reviewed_tracks import merge_reviewed_parts, split_reviewed_robots, validate_reviewed_provenance


class ReviewedPartsTests(unittest.TestCase):
    def parts(self):
        return [dict(track_id=i,identity_segment=f'{i}:100',team='462S',confidence=.6,
                     xyxy=[.4,.1+j*.14,.5,.27+j*.14]) for j,i in enumerate((111,105,88))]

    def group(self):
        return [dict(team='462S',segments=['111:100','105:100','88:100'],preferred_id=88,mode='union')]

    def test_three_parts_become_one_and_preserve_source_ids(self):
        parts=self.parts();out=merge_reviewed_parts(parts,self.group())
        self.assertEqual(len(out),1);self.assertEqual(out[0]['source_track_ids'],[88,105,111])
        self.assertEqual(out[0]['track_id'],88);self.assertAlmostEqual(out[0]['xyxy'][3],.55)
        self.assertEqual(len(parts),3)

    def test_reused_id_and_other_team_are_not_silently_merged(self):
        parts=self.parts();parts[-1]['identity_segment']='88:900'
        self.assertEqual(len(merge_reviewed_parts(parts,self.group())),2)
        parts=self.parts();parts[0]['team']='2899A'
        with self.assertRaises(ValueError):merge_reviewed_parts(parts,self.group())

    def test_widely_separated_parts_are_not_joined(self):
        parts=self.parts();parts[0]['xyxy']=[.05,.1,.1,.2]
        self.assertEqual(len(merge_reviewed_parts(parts,self.group())),3)


class ReviewedSplitTests(unittest.TestCase):
    def sample(self):
        original=[dict(track_id=262,identity_segment='262:3821',team=None,
            confidence=.85,xyxy=[.5,.2,.6,.4],identity_review_label='overlap')]
        review=dict(review_id='split-262',segment='262:3821',parent_track_id=262,
            parts=[dict(track_id='262a',team='2499B',xyxy=[.5,.2,.55,.34],pixel_correlation=.99),
                   dict(track_id='262b',team='88S',xyxy=[.53,.21,.6,.4],pixel_correlation=.99)])
        return original,review,{'2499B':'blue','88S':'red'}

    def test_two_children_keep_one_parent_and_frame_scoping(self):
        original,review,roster=self.sample()
        out=split_reviewed_robots(original,[review],roster)
        self.assertEqual([o['team'] for o in out],['2499B','88S'])
        self.assertTrue(all('identity_review_label' not in o for o in out))
        validate_reviewed_provenance(out,original,[review])
        self.assertEqual(split_reviewed_robots(original,[],roster),original)
        other=[dict(original[0],identity_segment='262:4000')]
        self.assertEqual(split_reviewed_robots(other,[review],roster),other)
        with self.assertRaises(ValueError):validate_reviewed_provenance(out[:1],original,[review])
        with self.assertRaises(ValueError):validate_reviewed_provenance(out+original,original,[review])

    def test_known_peer_and_invalid_box_cannot_be_overridden(self):
        original,review,roster=self.sample()
        with self.assertRaises(ValueError):
            split_reviewed_robots(original+[dict(original[0],track_id=215,team='2499B')],[review],roster)
        review['parts'][0]['xyxy']=[.5,.4,.4,.3]
        with self.assertRaises(ValueError):split_reviewed_robots(original,[review],roster)

    def test_split_ids_work_with_team_memory(self):
        from match_state import TeamMemory
        original,review,roster=self.sample()
        roster.update({'462S':'blue','2899A':'red'})
        out=split_reviewed_robots(original,[review],roster)
        memory=TeamMemory(roster,prediction_seconds=0)
        states=memory.update(131.54,out)
        self.assertEqual({s['team'] for s in states if s['status']=='observed'},{'2499B','88S'})
        memory.update(131.58,out)


if __name__=='__main__':unittest.main()
