"""Math stars are not Markdown emphasis; currency and literal code stay distinct."""
import unittest

from check_emphasis import _neutralize_code, check_emphasis


class EmphasisMathTests(unittest.TestCase):
    def test_single_dollar_math_stars_are_neutralized(self):
        for formula in (r"$a^*$", r"$a * b$", r"$2*x$", r"$W_{ij}^{*}$"):
            with self.subTest(formula=formula):
                self.assertEqual(_neutralize_code(formula), formula.replace("*", "+"))
        self.assertEqual(check_emphasis(r"最优解 $a^*$ 与 $b^*$。"), [])

    def test_display_math_stays_supported(self):
        self.assertEqual(_neutralize_code(r"$$a^* + b^*$$"), r"$$a^+ + b^+$$")

    def test_prices_do_not_hide_real_emphasis(self):
        for text in ("价格 $25，**（错误）**正文，另 $30。",
                     "价格 $25**（错误）**正文$30。",
                     r"价格 \$25，**（错误）**正文，另 \$30。"):
            with self.subTest(text=text):
                self.assertEqual(_neutralize_code(text), text)
                self.assertTrue(check_emphasis(text))

    def test_escaped_delimiters_are_literal(self):
        for text in (r"\$a^*\$", r"\$a^*$", r"$a^*\$"):
            self.assertEqual(_neutralize_code(text), text)

    def test_inline_code_dollars_do_not_pair_with_prose(self):
        text = "代码 `a$` **（错误）**正文 $x^*$。"
        self.assertEqual(_neutralize_code(text), "代码 `a$` **（错误）**正文 $x^+$。")
        self.assertTrue(check_emphasis(text))
        self.assertEqual(_neutralize_code("``a`*b`` 与 $x^*$"), "``a`+b`` 与 $x^+$")

    def test_regular_markdown_emphasis_is_not_changed(self):
        text = "正确的 **重点**，以及 **（错误）**正文。"
        self.assertEqual(_neutralize_code(text), text)
        self.assertTrue(check_emphasis(text))


if __name__ == "__main__":
    unittest.main()
