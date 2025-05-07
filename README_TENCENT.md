# Sphinx-Tencent


### Environment Configuration
```bash
conda create -n sphinx python=3.11.8
conda activate sphinx
pip install -r requirements.txt
```

### Task Ground Truth Format

`task_info.json` 中存储了三个 task 的例子，其中 id 要求互不相同，description 为 task 要完成的目标，apps 仅填 wechat 即可。

对于每个任务，需要收集的内容示例见 `./groundtruth/2/wechat`，即第二个任务 "send message 'test' to one of my friends"。其中：
- `meta.json` 中存储了四个动作，分别为打开聊天，在输入框输入 test，点击发送，任务结束。
- 每个动作前收集一次截图和 adb dump 出的 UI hierarchy。因为最后一个动作是 stop，无需再次收集。见 `x.xml` 和 `x.png`。
- `activities.json` 中存储了每个动作前收集的 activity。
- `evaluator.json` 中存储了用来 evaluate 一个 UI 序列和动作序列是否达成要求的 evaluator。

对于 ground truth，仅 evaluator 是必须收集的，其他内容在实际 agent 评测中没有用到。可以通过 `collect.py` 进行完整的收集，也可以直接手动写一下。

### Action 格式

- CLICK, SWIPE, LONGCLICK：需要给定 element 或者一个坐标
- INPUT：需要给定一个 element 或者一个坐标；需要给定 message 为输入的内容
- BACK, ENTER, STOP

### evaluator 格式

- 定义：
   1. match_rules：为一个 dict，这个 dict 的 key 为 action 或者 element 的 attribute，value 为这个 attribute 具有的信息。具体过程见下一行。
   2. match_type：取值为 equal 或者 include。equal 的意思是要求二者相等，include 的意思是要求 attribute 含有 check_rules 的 value 中的信息。Evaluator 需要能够根据 match_rules 和 match_type 的指定定位到某个 element（对于 stoppage 类型的 evaluator）或者某个 action（对于 findaction 类型的 evaluator）。
   3. check_rules：为一个 dict，这个 dict 的 key 为 action 或者 element 的 attribute，value 为这个 attribute 具有的信息。具体过程见下一行。
   4. check_type：取值为 equal 或者 include。equal 的意思是要求二者相等，include 的意思是要求 attribute 含有 check_rules 的 value 中的信息。在 Evaluator 获取需要比较的 element/action 之后，需要能够根据 check_rules 和 check_type 的指定比较某些 attribute 是否合法。
- 步骤：
   1. 输入 evaluator 类型：
      - stoppage：在最终页面（即 stop 动作对应的页面）中通过 match_type, match_rules 定位某个 element，并使用 check_type, check_rules 检验该 element 是否合法。比较特别的，check_rules 中可以使用 activity。
      - findaction：使用 match_type, match_rules 定位 action sequence 中的某个 action（注意这里既可以使用 action 的 attribute 如 action type 和 message，对于需要与 element 做交互的 action 也可以使用 element 的 attribute 如 resource-id, content-desc, class），并使用 check_type, check_rules 检验该 action 是否合法（同理，也可以使用 action 和 element 二者的 attribute）。
      - lastaction：检验最后一个动作是否满足一些条件，仅需要提供 check_type 和 check_rules。同上可以使用 action 和交互的 element 二者的 attribute。
      - findelement：使用 match_type, match_rules 定位 hierarchy sequence 中的某个 element，并使用 check_type, check_rules 检验该 element 是否合法。match_rules 和 check_rules 均可以使用 activity。
      - findelementbyaction：使用 action_match_type, action_match_rules 定位 action sequence 中的某个 action，并在该 action 对应的 hierarchy 上使用 element_match_type, element_match_rules 定位某个 element，使用 check_type, check_rules 检验该 element 是否合法。
   2. 按照脚本要求，根据不同的 evaluator 类型输入不同的参数。特别的，如果 check_type 和 match_type 为 equal，可以直接输入回车，脚本会自动选择 equal 类型。
   3. 输入参数后继续收集下一个 evaluator，如果没有 evaluator 直接输入回车结束 evaluator 收集。
   4. 在收集完所有 evaluator 后，脚本会自动使用收集的 evaluator 对收集的 trace 做评测。
- 原则：
   1. 对于一个 trace，不同 evaluator 之间的关系是 and，即如果一个 trace 能够通过多个 evaluator 才算成功。
   2. evaluator 的收集过程中应尽量避免 bounds 的使用，这是因为 bounds 是容易改变的。
   3. 如果对于一个 trace 有多个 evaluator，默认的实现是不考虑顺序的，也就是我们只关心整个序列中是否存在分别某个 action 或者某个 element，而不关心他们的先后，但可能一些情况下这些先后关系是重要的。如果遇到了这种情况，我们还提供了类型为 "rule" 的 evaluator，具体来说需要自行更改 evaluator.json 文件并在修改完成后重新跑一次脚本以检验更新后的 evaluator 是否正确。具体来说 rule 类型的 evaluator 要求传入一个 order 和一系列 evaluators，之后根据 order 类型检验 evaluators 是否合法。我们提供的 order 类型分别是 sequential, consecutive 和 present，其中 sequential 要求所有的 evaluator 在 trace 上进行顺序的匹配，consecutive 要求所有的 evaluator 在 trace 上匹配为一个连续（为了方便，我们将连续定义为既可以匹配 hierarchy 和后续的 action，也可以忽略中间的 hierarchy 匹配两个连续的 action）的子串，present 则与默认相同仅要求均出现。同时，我们支持 rule evaluator 进行嵌套。值得注意的是，我们在三种 order 类型中都允许了同一个 hierarchy 或者 action 被多个 evaluator 匹配，因此我们可以对同一个 hierarchy 或者 action 使用多个 evaluator。我们在 [groundtruth/901/a53](https://github.com/PKU-ASE-RISE/Sphinx/tree/main/groundtruth/901/a53) 文件夹下提供了 5 个可以通过测试的 evaluator 和 2 个不能通过测试的 evaluator 作为示范。

### Agent 运行

`run_benchmark.py` 实现了针对单个 task 的运行，仅支持了 ReAct Agent。支持了 text（即抽取出所有 clickable 的控件）, tree（对 UI 控件树进行了树形的描述）, image（纯截图）, annotated_image（即 Set-of-mark, SoM）四种 observation mode。
根据经验来看，对于通用模型，比较推荐尝试 text, tree, annotated_image。
对于 Qwen-VL, Kimi-V 或者 UI 模型（例如 UI-tars, OS-Atlas），比较推荐使用纯截图。

如果需要支持其他 agent，只需要在 agent 运行过程中记录 UI hierarchy 序列，UI 动作序列和 activity 序列（如果不需要 activity，可以随便填一下）即可。我们之前已经将 AppAgent 和 MobileAgent-v2 在 Sphinx 上运行过。