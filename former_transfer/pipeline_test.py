import unittest
import sentiment_analysis.sentiment_dictionary as sd
import article_selection.article_selection as arts
import sentiment_analysis.bert as bert
import os

class TestSentimentDictionary(unittest.TestCase):
    def test_wrong_input(self):
        text = "Flüchtlinge haben leider ein schlechtes Image."
        self.assertRaises(TypeError, sd.analyse_sentiment,text, [])
        self.assertRaises(TypeError, sd.analyse_sentiment,text, [2,3,4,"hallo"])
        self.assertRaises(TypeError, sd.analyse_sentiment,3, ["c"])
        self.assertRaises(TypeError, sd.analyse_sentiment,[text], ["c"])
        self.assertRaises(TypeError, sd.analyse_sentiment,True, ["c"])

    def test_running(self):
        sd.test()


class TestSentimentBert(unittest.TestCase):
    def test_running(self):
        bert.test()


class TestArticleSelection(unittest.TestCase):
    def test_wrong_input_is_topic_relevant(self):
        self.assertRaises(TypeError,arts.is_topic_relevant,"test")
        self.assertRaises(TypeError,arts.is_topic_relevant,3)
        self.assertRaises(TypeError,arts.is_topic_relevant,False)
   
        valid_article = {'date':'01.01.2020','title':"title",'text':"texttext", 'url':"http//a.de"}

        self.assertRaises(TypeError,arts.is_topic_relevant,valid_article,3)
        self.assertRaises(TypeError,arts.is_topic_relevant,valid_article,True)
        self.assertRaises(TypeError,arts.is_topic_relevant,valid_article,[])
        self.assertRaises(TypeError,arts.is_topic_relevant,valid_article,[2,3,5])

    def test_right_output_is_topic_relevant(self):
        self.assertEqual(arts.is_topic_relevant({"test":3}),False)

        valid_article = {'date':'01.01.2020','title':"title",'text':"texttext", 'url':"http//a.de"}
        self.assertEqual(arts.is_topic_relevant(valid_article),False)
        
        valid_list = ["text"]
        self.assertEqual(arts.is_topic_relevant(valid_article,valid_list),True)

        article_missing_date = {'title':"title",'text':"texttext", 'url':"http//a.de"}
        article_missing_title = {'date':'01.01.2020','text':"texttext", 'url':"http//a.de"}
        article_missing_text = {'date':'01.01.2020','title':"title", 'url':"http//a.de"}
        article_missing_url = {'date':'01.01.2020','title':"title",'text':"texttext"}
        self.assertEqual(arts.is_topic_relevant(article_missing_date,valid_list),False)
        self.assertEqual(arts.is_topic_relevant(article_missing_title,valid_list),False)
        self.assertEqual(arts.is_topic_relevant(article_missing_text,valid_list),False)
        self.assertEqual(arts.is_topic_relevant(article_missing_url,valid_list),False)

    def test_wrong_input_write_relevant_content_to_file(self):
        file_list = ["data/spiegel.json", "data/test.json"]
        relevant_articles_base = "base"
        search_keywords = ["test"]
        arts.write_relevant_content_to_file(file_list, relevant_articles_base, search_keywords)
        os.system("rm base_evaluation.json")

if __name__ == '__main__':
    unittest.main()