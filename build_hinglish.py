#!/usr/bin/env python3
"""Attach hand-authored Hinglish translations (by Claude, the LLM under test's
comparison point) to the 100-query gold set from queries_en.json.
Writes hinglish_queries.json: list of {gold, en, hi}.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "queries_en.json"), encoding="utf-8") as f:
    queries = json.load(f)

HI = {
122: "total income nikaalte time konsi deductions milti hain?",
123: "life insurance premium, deferred annuity aur PF contribution pe kitni tax deduction milti hai?",
125: "Agnipath Scheme mein contribution ke liye tax relief kaise milta hai?",
126: "health insurance premium pe tax relief kaise claim karein?",
128: "medical treatment ke expenses pe tax relief kaise milta hai?",
129: "education loan ke interest pe tax relief kaise milta hai?",
130: "apne rehne wale ghar ke home loan interest pe tax relief kaise milta hai?",
131: "kisi property ke liye liye gaye loan ke interest pe tax relief kaise milta hai?",
132: "electric vehicle kharidne pe tax relief kaise milta hai?",
133: "charitable donation pe tax relief kaise claim karein?",
134: "rent paid pe kya deduction milta hai?",
136: "companies dwara political party ko diye gaye contribution pe tax relief kaise milta hai?",
137: "kisi bhi vyakti dwara political party ko diya gaya contribution pe tax relief kaise milta hai?",
108: "same head ke income mein losses kaise adjust kar sakte hain?",
109: "kisi aur head ke income mein losses kaise adjust kar sakte hain?",
110: "house property se hua loss future mein kaise carry forward hota hai?",
111: "capital gains se hue loss ka carry forward kaise hota hai?",
112: "business loss ka carry forward kaise hota hai?",
114: "specified business ke losses ka set off aur carry forward kaise hota hai?",
115: "specified activity se hue losses ka set off aur carry forward kaise hota hai?",
119: "kin cases mein losses ka set off allowed nahi hota?",
120: "search ya survey ke baad undisclosed income ke against losses set off kyun nahi hote?",
6: "residential status kaise decide hota hai?",
18: "salary ke badle mile profits pe tax kaise lagta hai?",
19: "salary income se kya deduction milta hai?",
20: "house property se income kaise calculate hoti hai?",
22: "house property income se kya deduction milta hai?",
26: "business ya profession se hone wali income kaise calculate hoti hai?",
27: "business profit nikalne ka tarika kya hai?",
33: "depreciation pe tax benefit kaise milta hai?",
67: "capital gains ke baare mein Act kya kehta hai?",
72: "capital gains kaise calculate kiye jaate hain?",
82: "rehne wale property ko bechne pe hue profit ke baare mein kya niyam hai?",
85: "kuch bonds mein invest karne pe capital gains tax se chhoot milti hai kya?",
86: "residential house mein investment karne se kya tax benefit milta hai?",
99: "spouse aur minor child ki income, individual ki income mein kaise include hoti hai?",
224: "investment fund aur uske unit holders pe tax kaise lagta hai?",
214: "investment income aur long-term capital gains pe tax kaise lagta hai?",
397: "compliance aur reporting ke baare mein Act mein kya likha hai?",
461: "statements na dene par kya penalty lagti hai?",
57: "construction aur service contracts mein revenue recognition kaise hoti hai?",
222: "venture capital undertakings ki income pe tax kaise lagta hai?",
465: "returns ya statements ka inspection allow karne ke baare mein Act kya kehta hai?",
398: "TDS deduct ya pay na karne pe kya consequences hote hain?",
199: "kuch manufacturing domestic companies pe tax kaise lagta hai?",
142: "housing projects ke profits pe kya deduction milta hai?",
281: "section 280 ke under notice jaari karne se pehle kya procedure follow hota hai?",
54: "mineral oil dhundhne wale business ke baare mein Act kya kehta hai?",
183: "yeh Chapter kis par apply hota hai?",
329: "firm ka tax na chukane par partners ki joint liability kaise hoti hai?",
84: "kin cases mein yeh charge lagta hai?",
150: "federal co-operative society ki income pe kya deduction milta hai?",
407: "Assessing Officer ke order ke hisaab se advance tax kaise pay karte hain?",
75: "depreciable asset ka cost of acquisition kaise nikala jaata hai?",
476: "Chapter XIX-B ke under tax Central Government ko credit na karne pe kya hota hai?",
400: "Central Government is Chapter ke provisions ko relax karne ka authority kaise rakhti hai?",
297: "kuch cases mein interest ya penalty kyun nahi lagti?",
500: "revenue protect karne ke liye provisional attachment kaise hoti hai?",
260: "information ka faceless collection kaise hota hai?",
434: "TDS deduct karne ki liability deny karne pe refund kaise milta hai?",
519: "prosecution se immunity dene ka authority kise hai?",
7: "income ko received maana jaana aur dividend ko tax year mein income maana jaana, iske baare mein Act kya kehta hai?",
201: "new manufacturing domestic companies ki income pe tax kaise lagta hai?",
328: "ek firm ka doosri firm dwara succession kaise hota hai?",
30: "kuch premiums pe kya deduction milta hai?",
218: "services centre unit ke baare mein Act kya kehta hai?",
194: "1961 Act ke section 115B, 115BB jaise sections ke baare mein kya provision hai?",
21: "annual value kaise determine hoti hai?",
296: "block assessment complete karne ki time-limit kya hai?",
17: "Banking Regulation Act, 1949 ke Part V mein iska kya zikar hai?",
417: "State Government ke through tax recovery kaise hoti hai?",
124: "Central Government ki scheme ke baare mein Act kya kehta hai?",
79: "quoted share ke alawa doosre shares ke baare mein kya niyam hai?",
307: "jab beneficiaries ka pata na ho tab tax kaise charge hota hai?",
280: "income escape hone par assessment ke liye notice kaise issue hota hai?",
100: "kisi doosre vyakti ki income mein include hone wali income ke liye liability kiski hoti hai?",
91: "Valuation Officer ko reference kaise kiya jaata hai?",
382: "vacancy hone se proceedings invalid to nahi ho jaate?",
61: "non-residents ki kuch business activities ke baare mein kya rule hai?",
45: "scientific research pe hue expenditure ka kya tax treatment hai?",
182: "connected person aur accommodating party ka treatment kaise hota hai?",
88: "urban area se Special Economic Zone mein undertaking shift karne pe kya provision hai?",
28: "rent, rates, taxes, repairs aur insurance ka kya treatment hota hai?",
326: "section 325 comply na hone par assessment kaise hoti hai?",
141: "kuch industrial undertakings ke profits pe tax relief kaise milta hai?",
12: "political parties aur electoral trusts ki kaunsi income total income mein include nahi hoti?",
444: "books of account mein false entry karne pe kya penalty lagti hai?",
381: "Board for Advance Rulings ke baare mein Act kya kehta hai?",
367: "Supreme Court mein appeal kaise ki jaati hai?",
153: "deposits ke interest pe tax benefit kaise milta hai?",
40: "kuch assets ka cost of acquisition kaise compute hota hai?",
254: "kuch information collect karne ka authority kise hai?",
96: "assets transfer kiye bina income transfer karne pe kya rule hai?",
243: "cases transfer karne ka authority kise hai?",
231: "tonnage tax scheme opt karne ka method aur uski validity kya hai?",
204: "new manufacturing co-operative societies ki income pe tax kaise lagta hai?",
505: "liaison office rakhne wale non-resident ko statement kaise submit karna hota hai?",
523: "kin circumstances mein notice ko valid mana jaata hai?",
247: "search aur seizure ke baare mein Act kya kehta hai?",
344: "property ke roop mein rakhi gayi business undertaking ke baare mein kya provision hai?",
}

out = []
missing = []
for row in queries:
    g = row["gold"]
    if g not in HI:
        missing.append(g)
        continue
    out.append({"gold": g, "en": row["en"], "hi": HI[g]})

if missing:
    print("MISSING HI for golds:", missing)
    raise SystemExit(1)

out_path = os.path.join(HERE, "hinglish_queries.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"Wrote {len(out)} rows to {out_path}")
