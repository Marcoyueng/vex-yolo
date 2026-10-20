import unittest
from types import SimpleNamespace
from replay_timeline import ReplayTimeline,timecode


class TimelineTests(unittest.TestCase):
    def test_scrub_coordinates_follow_zoomed_range_and_clamp_to_clip(self):
        timeline=SimpleNamespace(LEFT=78,total=3001,first=900,span=lambda:300,width=lambda:600)
        self.assertEqual(ReplayTimeline.at(timeline,378),1050)
        self.assertEqual(ReplayTimeline.at(timeline,-5000),0)
        self.assertEqual(ReplayTimeline.at(timeline,10000),3000)

    def test_zoom_keeps_pointer_on_same_frame(self):
        timeline=SimpleNamespace(frame=500,total=3001,first=0.,zoom=1.,paint=lambda:None)
        timeline.span=lambda:(timeline.total-1)/timeline.zoom
        before=(500-timeline.first)/timeline.span()
        ReplayTimeline.set_zoom(timeline,2,500)
        self.assertEqual((500-timeline.first)/timeline.span(),before)
        self.assertEqual(timecode(1800,30),'00:01:00:00')


if __name__=='__main__':unittest.main()
